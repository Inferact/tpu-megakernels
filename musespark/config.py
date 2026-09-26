"""Muse Spark 1.2 text-decoder configuration.

`Config` is the single source of truth for every shape in the package: the reference
model (`musespark`), the per-rank kernel layout (`musespark.layout`) and the kernels all
derive their sizes from it, so a small `MINI` config runs the identical code paths.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    layers: int = 62
    hidden: int = 8192  # H: residual stream width
    moe_hidden: int = 4096  # Hm: expert input/output width (pre/post_expert_proj around it)
    expert_hidden: int = 4096  # I: expert FFN width (gate_up_proj is [Hm, 2I], down is [I, Hm])
    experts: int = 256  # E
    top_k: int = 8
    heads: int = 128
    kv_heads: int = 16
    head_dim: int = 64
    vocab: int = 202048  # embedding / lm_head rows
    vocab_used: int = 201818  # ids >= vocab_used never occur (untrained padding rows)
    sliding_window: int = 2048  # keys per query INCLUDING the query token itself
    rope_theta: float = 500000.0
    rms_eps: float = 1e-5  # every pre-norm, the QK norms and the attention-output norm
    post_eps: float = 1e-8  # every post-norm (post_attn, post_ffn, post_expert)
    gate_temperature: float = 0.3  # residual gate logits are divided by this
    qk_scale_factor: float = 5.4730064863875  # softmax scale = qk_scale_factor / sqrt(head_dim)
    output_multiplier: float = 0.17677669529663687  # 2**-2.5: router logits AND final logits
    softcap: float = 20.0  # logits = softcap * tanh(logits / softcap)
    route_eps: float = 1e-15  # top-k weight renormalisation: w / (sum(w) + route_eps)
    group_size: int = 128  # int4 quantization group along the contraction axis
    eos: tuple[int, ...] = (200001, 200008)  # <|end_of_text|>, <|eot|>
    bos: int = 200000
    pad: int = 200018
    full_attention_every: int = 4  # full-attention (NoPE) layers are l % every == offset
    full_attention_offset: int = 1

    def __post_init__(self):
        if self.heads * self.head_dim != self.hidden:
            raise ValueError("Muse Spark requires heads * head_dim == hidden")
        if self.heads % self.kv_heads:
            raise ValueError("heads must be a multiple of kv_heads")
        if self.vocab_used > self.vocab:
            raise ValueError("vocab_used must not exceed vocab")

    @classmethod
    def from_checkpoint(cls, directory, vocab_used=201818):
        """Read `config.json` (`text_config`) and assert every assumption of the spec."""
        directory = Path(directory)
        raw = json.loads((directory / "config.json").read_text())
        c = raw.get("text_config", raw)
        required = {
            "attention_bias": False,
            "hidden_act": "silu",
            "num_shared_experts": 0,
            "routed_scaling_factor": 1.0,
            "tie_word_embeddings": False,
        }
        for name, expected in required.items():
            if c[name] != expected:
                raise ValueError(f"Unsupported Muse Spark setting {name}={c[name]!r}")
        rope = c["rope_parameters"]
        if rope["rope_type"] != "default":
            raise ValueError(f"Unsupported Muse Spark rope {rope!r}")
        layers = c["num_hidden_layers"]
        if set(c["mlp_layer_types"]) != {"sparse"}:
            raise ValueError("Muse Spark requires every layer to be sparse (MoE)")
        every, offset = 4, 1
        want = tuple(
            "full_attention" if i % every == offset else "sliding_attention" for i in range(layers)
        )
        if tuple(c["layer_types"]) != want:
            raise ValueError("Muse Spark layer pattern must be S F S S repeating")
        theta = rope["rope_theta"]
        want_theta = tuple(0.0 if i % every == offset else theta for i in range(layers))
        if tuple(float(t) for t in c["layer_rope_theta"]) != want_theta:
            raise ValueError("layer_rope_theta must be 0 exactly on the full-attention layers")
        eos = (c["eos_token_id"],)
        gen = directory / "generation_config.json"
        if gen.exists():
            g = json.loads(gen.read_text()).get("eos_token_id", c["eos_token_id"])
            eos = tuple(g) if isinstance(g, list) else (g,)
        return cls(
            layers=layers,
            hidden=c["hidden_size"],
            moe_hidden=c["moe_hidden_size"],
            expert_hidden=c["moe_intermediate_size"],
            experts=c["num_local_experts"],
            top_k=c["num_experts_per_tok"],
            heads=c["num_attention_heads"],
            kv_heads=c["num_key_value_heads"],
            head_dim=c["head_dim"],
            vocab=c["vocab_size"],
            vocab_used=vocab_used,
            sliding_window=c["sliding_window"],
            rope_theta=theta,
            rms_eps=c["rms_norm_eps"],
            post_eps=c["post_norm_eps"],
            gate_temperature=c["residual_gate_temperature"],
            qk_scale_factor=c["qk_scale_factor"],
            output_multiplier=c["output_multiplier"],
            softcap=c["final_logit_softcapping"],
            eos=eos,
            bos=c["bos_token_id"],
            pad=c["pad_token_id"],
            full_attention_every=every,
            full_attention_offset=offset,
        )

    # ---- derived quantities -------------------------------------------------------------
    def is_full_attention(self, layer):
        """True on the full-attention layers, which are also the NoPE layers."""
        return layer % self.full_attention_every == self.full_attention_offset

    @property
    def full_layers(self):
        return tuple(i for i in range(self.layers) if self.is_full_attention(i))

    @property
    def sliding_layers(self):
        return tuple(i for i in range(self.layers) if not self.is_full_attention(i))

    @property
    def q_width(self):
        return self.heads * self.head_dim

    @property
    def kv_width(self):
        return self.kv_heads * self.head_dim

    @property
    def group_size_q(self):
        """Query heads per KV head (GQA group)."""
        return self.heads // self.kv_heads

    @property
    def softmax_scale(self):
        return self.qk_scale_factor / self.head_dim**0.5


# Test configuration (design.md section 6): per rank with tp=8 -> 2 q heads / 1 kv head,
# expert slices gate_up [512, 2*64] / down [64, 512]; the group size must divide 64.
MINI = Config(
    layers=4,
    hidden=1024,
    moe_hidden=512,
    expert_hidden=512,
    experts=16,
    top_k=4,
    heads=16,
    kv_heads=8,
    head_dim=64,
    vocab=2048,
    vocab_used=2048,
    sliding_window=256,
    group_size=64,
    eos=(1, 2),
    bos=0,
    pad=3,
)
