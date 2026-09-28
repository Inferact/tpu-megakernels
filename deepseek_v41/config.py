"""Architecture configuration and layer-kind helpers for DeepSeek-V4.1-Flash."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class LayerKind(str, Enum):
    """Distinct attention / state-sharing layer kinds in DeepSeek-V4.1-Flash."""

    SWA = "swa"                       # L0-1: sliding window only (r=0)
    FULL_R2 = "full_r2"               # L2, 8, 14: r=2 compressor + owner indexer
    REUSE_R2 = "reuse_r2"             # L3-7, 9-13, 15-19: reuse latest r=2 KV & top-k
    FULL_R1_CAND = "full_r1_cand"     # L20: r=1 compressor + owner indexer + 2048x8 candidates
    REINDEX_R1 = "reindex_r1"         # L24, 28, 32, 36: r=1 reindexer inside L20 candidates
    REUSE_R1 = "reuse_r1"             # L21-23, 25-27, 29-31, 33-35, 37-39: reuse L20 KV + latest r=1 top-k
    DSPARK = "dspark"                 # L40-42 (mtp.0..2): DSpark speculative draft block


@dataclass(frozen=True)
class DSV41Config:
    """Complete model and decode configuration matching official inference/config.json."""

    vocab_size: int = 129280
    padded_vocab_size: int = 131072
    dim: int = 5120
    n_layers: int = 40
    n_mtp_layers: int = 3
    n_heads: int = 64
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128
    compress_ratios: tuple[int, ...] = (
        0, 0,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0,
    )
    kv_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0
    original_seq_len: int = 65536
    beta_fast: int = 32
    beta_slow: int = 1
    n_routed_experts: int = 384
    n_activated_experts: int = 6
    n_shared_experts: int = 1
    moe_inter_dim: int = 2304
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    dspark_n_routed_experts: int = 128
    dspark_n_activated_experts: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: tuple[int, ...] = (37, 38, 39)
    markov_rank: int = 256
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-20
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_max_ngram_size: int = 4
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_vocab_size: int = 16000000
    engram_num_embeddings: tuple[int, ...] = (384006168, 384016682)
    engram_pad_id: int = 2
    index_k_mode: str = "reference"  # "reference" (model.py:537-554) or "intended"

    def layer_kind(self, layer_id: int) -> LayerKind:
        """Returns the LayerKind for backbone (0..n_layers-1) or DSpark (n_layers..n_layers+n_mtp_layers-1)."""
        if layer_id >= self.n_layers:
            return LayerKind.DSPARK
        r = self.compress_ratios[layer_id]
        if r == 0:
            return LayerKind.SWA
        if layer_id == self.candidate_source_layer_id:
            return LayerKind.FULL_R1_CAND
        if layer_id in self.kv_source_layer_ids:
            return LayerKind.FULL_R2 if r == 2 else LayerKind.FULL_R1_CAND
        if layer_id in self.index_source_layer_ids:
            return LayerKind.REINDEX_R1
        return LayerKind.REUSE_R2 if r == 2 else LayerKind.REUSE_R1

    def has_engram(self, layer_id: int) -> bool:
        return layer_id in self.engram_layer_ids

    def kv_source_for(self, layer_id: int) -> int | None:
        """Returns the layer_id whose compress_kv is consumed by layer_id, or None if r==0."""
        if layer_id >= self.n_layers or self.compress_ratios[layer_id] == 0:
            return None
        sources = [s for s in self.kv_source_layer_ids if s <= layer_id]
        return sources[-1] if sources else None

    def index_source_for(self, layer_id: int) -> int | None:
        """Returns the layer_id whose topk_idxs are consumed by layer_id, or None if r==0."""
        if layer_id >= self.n_layers or self.compress_ratios[layer_id] == 0:
            return None
        sources = [s for s in self.index_source_layer_ids if s <= layer_id]
        return sources[-1] if sources else None

    @classmethod
    def from_json(cls, path: str | Path, **overrides) -> DSV41Config:
        data = json.loads(Path(path).read_text())
        fields = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        for k, v in list(fields.items()):
            if isinstance(getattr(cls, k), tuple) and isinstance(v, list):
                fields[k] = tuple(v)
        fields.update(overrides)
        return cls(**fields)


def full_config(**overrides) -> DSV41Config:
    """Returns the full 40+3-layer DeepSeek-V4.1-Flash config."""
    return DSV41Config(**overrides)


def tiny_config(**overrides) -> DSV41Config:
    """Returns an 8+2-layer scaled-down config that exercises every LayerKind and Engram on CPU/interpret."""
    defaults = dict(
        vocab_size=1024,
        padded_vocab_size=1024,
        dim=256,
        n_layers=8,
        n_mtp_layers=2,
        n_heads=32,
        head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=8,
        sliding_window=16,
        # L0: SWA, L1: SWA+Engram, L2: FULL_R2, L3: REUSE_R2, L4: FULL_R2+Engram,
        # L5: FULL_R1_CAND, L6: REINDEX_R1, L7: REUSE_R1, L8-9: DSPARK
        compress_ratios=(0, 0, 2, 2, 2, 1, 1, 1, 0, 0),
        kv_source_layer_ids=(2, 4, 5),
        index_source_layer_ids=(2, 4, 5, 6),
        candidate_source_layer_id=5,
        candidate_topk_blocks=8,
        candidate_block_size=4,
        index_n_heads=32,
        index_head_dim=64,
        index_topk=8,
        n_routed_experts=32,
        n_activated_experts=6,
        n_shared_experts=1,
        moe_inter_dim=128,
        dspark_n_routed_experts=16,
        dspark_n_activated_experts=3,
        dspark_block_size=5,
        dspark_noise_token_id=1023,
        dspark_target_layer_ids=(5, 6, 7),
        markov_rank=64,
        engram_layer_ids=(1, 4),
        engram_n_heads=8,
        engram_head_dim=32,
        engram_vocab_size=2048,
        engram_num_embeddings=(49152, 49152),
        engram_pad_id=2,
    )
    defaults.update(overrides)
    return DSV41Config(**defaults)
