"""Load Kimi K3 checkpoint assets into the TP32 megakernel layout.

This module reads the tokenizer and Hugging Face checkpoint tensors, describes
the selected per-rank weight layout, and streams either raw or pre-sharded
weights onto the TPU. Routed gate/up experts are losslessly reblocked from
MXFP4 into the FP8 representation consumed by the decoder while loading.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ast
import base64
import json
import os
from pathlib import Path
import struct
import time
from typing import Callable

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding

from . import Config, unpack_mxfp4


BF16 = np.dtype(ml_dtypes.bfloat16)
PREFIX = "language_model.model."


def load_tokenizer(directory):
    """Read the checkpoint's tiktoken ranks, regex, and reserved-token mapping.

    Parse only the literal regex list from its tokenizer source; loading a
    tokenizer never executes checkpoint Python code or imports Transformers.
    """
    import tiktoken

    directory = Path(directory)
    config = json.loads((directory / "tokenizer_config.json").read_text())
    source = ast.parse((directory / "tokenization_kimi.py").read_text())
    cls = next(
        n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "TikTokenTokenizer"
    )
    assignment = next(
        n
        for n in cls.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pat_str" for t in n.targets)
    )
    value = assignment.value
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Constant)
        and value.func.value.value == "|"
        and value.func.attr == "join"
        and len(value.args) == 1
    ):
        raise ValueError("Unsupported checkpoint tokenizer pattern declaration")
    pattern = "|".join(ast.literal_eval(value.args[0]))
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (
            line.split() for line in (directory / "tiktoken.model").read_bytes().splitlines()
        )
    }
    added = config["added_tokens_decoder"]
    special = {
        added.get(str(i), {}).get("content", f"<|reserved_token_{i}|>"): i
        for i in range(len(ranks), len(ranks) + 256)
    }
    return tiktoken.Encoding(
        name="kimi-k3-checkpoint", pat_str=pattern, mergeable_ranks=ranks, special_tokens=special
    )


def pack_expert_bytes(value):
    """Checkpoint [N,K/2] bytes -> kernel [K/8,N] little-endian nibble groups."""
    if value.dtype != np.uint8 or value.ndim != 2 or value.shape[1] % 4:
        raise ValueError("Expected whole uint32 groups of packed FP4 bytes")
    return np.ascontiguousarray(value).view("<u4").T.copy()


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.config = Config.from_checkpoint(path)
        if self.config != Config():
            raise ValueError("This loader requires the full 93-layer/896-expert K3 contract")
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        missing = [f for f in set(self.index.values()) if not (self.path / f).is_file()]
        if missing:
            raise FileNotFoundError(f"Checkpoint is missing {len(missing)} tensor shards")

        self._layouts = {}

    def close(self):
        """Kept for callers that pair open/close; handles are per read."""
        return None

    # Coalesced streaming reads. On high-latency FUSE-backed storage, one file
    # handle streaming a contiguous range runs at ~600 MB/s (several handles in
    # parallel ~1 GB/s), a seek on a handle costs ~0.5 s, and interleaving
    # threads on one handle collapses to ~15 MiB/s. So a rank's 672 expert
    # tensors per layer are grouped into contiguous ranges (merging gaps up to
    # COALESCE_GAP, since wasted bytes are cheaper than a new stream), and each
    # range is streamed sequentially on its own handle, ranges in parallel.
    COALESCE_GAP = 16 << 20
    CHUNK = 16 << 20
    READ_THREADS = 8

    def _layout(self, filename):
        """(header dict, data start) of one shard, cached."""
        if filename not in self._layouts:
            with open(self.path / filename, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                self._layouts[filename] = (json.loads(f.read(size)), 8 + size)
        return self._layouts[filename]

    def read_many(self, keys):
        """Whole tensors for ``keys`` as uint8-viewable numpy arrays, via coalesced reads.

        Returns ``{key: array}`` with the header dtype/shape applied. Used for
        the expert families; a fake checkpoint without files (tests) falls
        back to ``read`` per key.
        """
        if not hasattr(self, "path"):
            return {key: self.read(key) for key in keys}
        by_file = {}
        for key in keys:
            by_file.setdefault(self.index[key], []).append(key)
        out = {}
        for filename, file_keys in by_file.items():
            header, base = self._layout(filename)
            spans = sorted((header[k]["data_offsets"][0], header[k]["data_offsets"][1], k) for k in file_keys)
            ranges = []
            for start, end, key in spans:
                if ranges and start - ranges[-1][1] <= self.COALESCE_GAP:
                    ranges[-1][1] = max(ranges[-1][1], end)
                    ranges[-1][2].append(key)
                else:
                    ranges.append([start, end, [key]])
            path = self.path / filename

            def stream(bounds):
                start, end, _ = bounds
                fd = os.open(path, os.O_RDONLY)
                try:
                    pieces = []
                    for offset in range(start, end, self.CHUNK):
                        pieces.append(os.pread(fd, min(self.CHUNK, end - offset), base + offset))
                finally:
                    os.close(fd)
                return b"".join(pieces)

            with ThreadPoolExecutor(min(self.READ_THREADS, len(ranges))) as pool:
                buffers = dict(enumerate(pool.map(stream, ranges)))
            for r, (start, end, range_keys) in enumerate(ranges):
                buffer = np.frombuffer(buffers[r], np.uint8)
                for key in range_keys:
                    meta = header[key]
                    a, b = meta["data_offsets"]
                    out[key] = buffer[a - start:b - start].view(self._DTYPES[meta["dtype"]]).reshape(meta["shape"])
        return out

    @staticmethod
    def _expert_parts(packed, scale, down, part, parts):
        if part is None:
            return packed, scale
        if part not in range(parts) or 3072 % parts:
            raise ValueError(f"Expert TP part must be in 0..{parts - 1}")
        width = 3072 // parts
        if down:
            return (
                packed[:, part * width // 2:(part + 1) * width // 2],
                scale[:, part * width // 32:(part + 1) * width // 32],
            )
        return packed[part * width:(part + 1) * width], scale[part * width:(part + 1) * width]

    def experts(self, layer, first, count, *, part=None, parts=4):
        """``(gate_up, gate_up_scales, down, down_scales)`` for ``count`` consecutive experts.

        Each element is a list over experts in kernel layout (see :meth:`expert`);
        all tensors of the layer's experts are fetched with :meth:`read_many`.
        """
        base = f"{PREFIX}layers.{layer}.block_sparse_moe.experts."
        keys = [
            f"{base}{first + e}.{name}.{kind}"
            for e in range(count)
            for name in ("w1", "w2", "w3")
            for kind in ("weight_packed", "weight_scale")
        ]
        tensors = self.read_many(keys)
        gate_up, gate_up_scales, downs, down_scales = [], [], [], []
        for e in range(count):
            prefix = f"{base}{first + e}."
            weights, scales = [], []
            for name in ("w1", "w3"):
                packed, scale = self._expert_parts(
                    tensors[prefix + name + ".weight_packed"], tensors[prefix + name + ".weight_scale"], False, part, parts
                )
                if scale.dtype != np.uint8 or np.any((scale == 0) | (scale == 255)):
                    raise ValueError(f"{prefix}{name}: expected normal, finite E8M0 scales")
                weights.append(pack_expert_bytes(packed))
                scales.append(scale.T.copy())
            gate_up.append(np.concatenate(weights, axis=1))
            gate_up_scales.append(np.concatenate(scales, axis=1))
            packed, scale = self._expert_parts(
                tensors[prefix + "w2.weight_packed"], tensors[prefix + "w2.weight_scale"], True, part, parts
            )
            if scale.dtype != np.uint8 or np.any((scale == 0) | (scale == 255)):
                raise ValueError(f"{prefix}w2: expected normal, finite E8M0 scales")
            downs.append(pack_expert_bytes(packed))
            down_scales.append(scale.T.copy())
        return gate_up, gate_up_scales, downs, down_scales

    _DTYPES = {"U8": np.uint8, "BF16": BF16, "F32": np.float32, "F16": np.float16}

    def read(self, key, selection=None):
        """One tensor, or ``tensor[selection]``, streamed from the shard.

        A leading-axis slice (``(rows, ...full...)`` or a bare slice) is a
        contiguous byte range and is streamed as such; any other selection
        reads the whole tensor and slices in memory. Both avoid small,
        non-contiguous reads, which can be slow on network-backed storage.
        """
        filename = self.index[key]
        header, base = self._layout(filename)
        meta = header[key]
        shape = list(meta["shape"])
        dtype = self._DTYPES[meta["dtype"]]
        start, end = meta["data_offsets"]
        rows = slice(None)
        rest = ()
        if selection is not None:
            parts = selection if isinstance(selection, tuple) else (selection,)
            leading, rest = parts[0], parts[1:]
            if isinstance(leading, slice) and all(
                isinstance(r, slice) and r == slice(None) for r in rest
            ):
                rows, rest = leading, ()
            else:
                rest = parts
        r0, r1, step = rows.indices(shape[0])
        if step != 1:
            raise ValueError("Only unit-step leading slices are supported")
        row_bytes = (end - start) // shape[0] if shape else end - start
        offset = base + start + r0 * row_bytes
        length = (r1 - r0) * row_bytes if shape else end - start
        fd = os.open(self.path / filename, os.O_RDONLY)
        try:
            pieces = []
            for chunk_start in range(0, length, self.CHUNK):
                pieces.append(os.pread(fd, min(self.CHUNK, length - chunk_start), offset + chunk_start))
        finally:
            os.close(fd)
        value = np.frombuffer(b"".join(pieces), dtype).reshape([r1 - r0, *shape[1:]] if shape else [])
        return value[rest] if rest else value

    def expert(self, layer, expert, down=False, part=None, parts=2):
        """One expert's packed FP4 weights and E8M0 scales, optionally one TP part.

        ``part`` selects the rank's slice of an expert split ``parts`` ways over
        its gate/up output channels (``down=False``) or its down contraction
        channels (``down=True``).
        """
        base = f"{PREFIX}layers.{layer}.block_sparse_moe.experts.{expert}."
        names = ("w2",) if down else ("w1", "w3")
        weights, scales = [], []
        for name in names:
            packed_slice = scale_slice = None
            if part is not None:
                if part not in range(parts) or 3072 % parts:
                    raise ValueError(f"Expert TP part must be in 0..{parts - 1}")
                width = 3072 // parts  # channels of one part
                packed_slice = (
                    (slice(None), slice(part * width // 2, (part + 1) * width // 2))
                    if down
                    else (slice(part * width, (part + 1) * width), slice(None))
                )
                scale_slice = (
                    (slice(None), slice(part * width // 32, (part + 1) * width // 32))
                    if down
                    else (slice(part * width, (part + 1) * width), slice(None))
                )
            # Read whole tensors and slice in memory: a column slice of a
            # memory-mapped [N, K/2] tensor is one strided read per row, and
            # Repeating those reads across every expert and layer can leave
            # ranks blocked on storage. Row slices would be contiguous, but
            # whole reads keep the pattern uniform and the four TP parts of an
            # expert share a host, so the page cache serves the other three.
            packed = self.read(base + name + ".weight_packed")
            scale = self.read(base + name + ".weight_scale")
            if packed_slice is not None:
                packed = packed[packed_slice]
                scale = scale[scale_slice]
            weights.append(pack_expert_bytes(packed))
            if scale.dtype != np.uint8 or np.any((scale == 0) | (scale == 255)):
                raise ValueError(f"{base}{name}: expected normal, finite E8M0 scales")
            scales.append(scale.T.copy())
        return np.concatenate(weights, axis=1), np.concatenate(scales, axis=1)

    def dense(self, name, layer, rank):
        """One layer of a parameter family, with its exact global-rank slice."""
        base = f"{PREFIX}layers.{layer}."
        rows = slice(rank * 384, (rank + 1) * 384)
        heads = slice(rank * 3, (rank + 1) * 3)

        def read(suffix, selection=None):
            return self.read(base + suffix, selection)

        def matrix(suffix, output=slice(None), contraction=slice(None)):
            return read(suffix + ".weight", (output, contraction)).T.copy()

        def folded(stem, global_output=False):
            pre = PREFIX if global_output else base
            return self.read(pre + stem + "_norm.weight").astype(np.float32) * self.read(
                pre + stem + "_proj.weight"
            ).astype(np.float32)

        if name in ("embedding", "lm_head"):
            key = (
                PREFIX + "embed_tokens.weight"
                if name == "embedding"
                else "language_model.lm_head.weight"
            )
            value = self.read(key, (slice(rank * 5120, (rank + 1) * 5120), slice(None)))
            return value if name == "embedding" else value.T.copy()
        if name == "final_norm":
            return self.read(PREFIX + "norm.weight")[None]
        if name == "output_res":
            return folded("output_attn_res", True)
        if name in ("attn_res", "mlp_res"):
            return folded("self_attention_res" if name == "attn_res" else "mlp_res")
        if name in ("attn_norm", "ffn_norm"):
            return read(
                ("input_layernorm" if name == "attn_norm" else "post_attention_layernorm")
                + ".weight"
            )[None]
        if name in ("dense_gu", "shared_gu"):
            stem, real, padded = (
                ("mlp", 1056, 1152)
                if name == "dense_gu"
                else ("block_sparse_moe.shared_experts", 192, 256)
            )
            selection = slice(rank * real, (rank + 1) * real)
            return np.concatenate(
                [
                    np.pad(
                        matrix(stem + "." + p + "_proj", selection), ((0, 0), (0, padded - real))
                    )
                    for p in ("gate", "up")
                ],
                axis=1,
            )
        if name in ("dense_down", "shared_down"):
            stem, real, padded = (
                ("mlp", 1056, 1152)
                if name == "dense_down"
                else ("block_sparse_moe.shared_experts", 192, 256)
            )
            return np.pad(
                matrix(stem + ".down_proj", contraction=slice(rank * real, (rank + 1) * real)),
                ((0, padded - real), (0, 0)),
            )
        if name == "router":
            return matrix("block_sparse_moe.gate")
        if name == "router_bias":
            return read("block_sparse_moe.gate.e_score_correction_bias")[None]
        if name == "latent_norm":
            return np.pad(
                read("block_sparse_moe.routed_expert_norm.weight").astype(np.float32)[None],
                ((0, 0), (0, 512)),
            )
        if name in ("latent_down", "latent_up"):
            shape = (7168, 128) if name == "latent_down" else (128, 7168)
            if rank >= 28:
                return np.zeros(shape, BF16)
            selection = slice(rank * 128, (rank + 1) * 128)
            stem = "block_sparse_moe.routed_expert_"
            return (
                matrix(stem + "down_proj", selection)
                if name == "latent_down"
                else matrix(stem + "up_proj", contraction=selection)
            )
        if name == "k_gate":
            # The gate columns of k_projection (1152:1280) as their own array;
            # the gate-first KDA schedule fetches them separately.
            return matrix("self_attn.f_a_proj")
        if name == "k_projection":
            # q | k | v | g | b (the gate columns f_a_proj are k_gate).
            parts = [matrix("self_attn." + p + "_proj", rows) for p in ("q", "k", "v")]
            parts += [matrix("self_attn.g_proj", rows)]
            parts += [np.pad(matrix("self_attn.b_proj", heads), ((0, 0), (0, 125)))]
            return np.concatenate(parts, axis=1)
        if name == "k_fb":
            return matrix("self_attn.f_b_proj", rows)
        if name in ("k_o", "m_o"):
            return matrix("self_attn.o_proj", contraction=rows)
        if name == "k_conv":
            return np.stack(
                [
                    read("self_attn." + p + "_conv1d.weight", (rows, slice(None), slice(None)))[
                        :, 0
                    ].T.reshape(4, 3, 128)
                    for p in ("q", "k", "v")
                ],
                axis=1,
            )
        if name == "k_a_log":
            return read("self_attn.A_log", heads).reshape(3, 1)
        if name == "k_dt":
            return read("self_attn.dt_bias", rows).reshape(3, 128)
        if name == "k_norm":
            return np.pad(read("self_attn.o_norm.weight")[None], ((0, 7), (0, 0)))
        if name == "m_qa":
            return matrix("self_attn.q_a_proj")
        if name == "m_ka":
            return np.pad(matrix("self_attn.kv_a_proj_with_mqa"), ((0, 0), (0, 64)))
        if name == "m_qb":
            value = matrix("self_attn.q_b_proj", slice(rank * 576, (rank + 1) * 576))
            return np.pad(value.reshape(1536, 3, 192), ((0, 0), (0, 0), (0, 64))).reshape(1536, 768)
        if name == "m_kb":
            return matrix("self_attn.kv_b_proj", slice(rank * 768, (rank + 1) * 768))
        if name == "m_gate":
            return matrix("self_attn.g_proj", rows)
        if name in ("m_qnorm", "m_knorm"):
            return read(
                "self_attn." + ("q_a" if name == "m_qnorm" else "kv_a") + "_layernorm.weight"
            )[None]
        raise KeyError(name)

    def dense_selected(self, name, layer, rank):
        """One layer of a parameter family in the *selected* megakernel layout.

        The selected TP32 layout differs from :meth:`dense` for four families;
        everything else is unchanged:

        * ``k_o``/``m_o``: paired-rank output projections. Rank ``r`` holds the
          contraction channels of ranks ``2p, 2p+1`` (``p = r // 2``, 768 rows)
          for output half ``r % 2`` -> ``[768, hidden / 2]``.
        * ``latent_up``: output-sharded over 8 blocks of 1024 hidden columns
          (block ``r % 8``; block 7 is zero padding past 7168), replicated
          across the four hosts, with the padded latent rows dropped ->
          ``[3584, 1024]``.
        * ``shared_gu``/``shared_down``: the rank's 192 shared channels without
          the 256 padding -> ``[hidden, 384]`` / ``[192, hidden]``.
        """
        base = f"{PREFIX}layers.{layer}."
        hidden = self.config.dim
        if name in ("k_o", "m_o"):
            pair, half = divmod(rank, 2)
            value = self.read(
                base + "self_attn.o_proj.weight",
                (slice(half * hidden // 2, (half + 1) * hidden // 2), slice(pair * 768, (pair + 1) * 768)),
            )
            return value.T.copy()
        if name == "latent_up":
            block = rank % 8
            if (block + 1) * 1024 > hidden:
                return np.zeros((self.config.latent, 1024), BF16)
            value = self.read(
                base + "block_sparse_moe.routed_expert_up_proj.weight",
                (slice(block * 1024, (block + 1) * 1024), slice(None)),
            )
            return value.T.copy()
        if name == "shared_gu":
            rows = slice(rank * 192, (rank + 1) * 192)
            return np.concatenate(
                [
                    self.read(
                        base + f"block_sparse_moe.shared_experts.{p}_proj.weight", (rows, slice(None))
                    ).T
                    for p in ("gate", "up")
                ],
                axis=1,
            )
        if name == "shared_down":
            return self.read(
                base + "block_sparse_moe.shared_experts.down_proj.weight",
                (slice(None), slice(rank * 192, (rank + 1) * 192)),
            ).T.copy()
        return self.dense(name, layer, rank)


def selected_local_shapes(layers=93, hidden=7168, *, vocab=163840):
    """Per-rank shapes of the selected (golden-validated) megakernel layout.

    This is the clean decoder's TP4/EP8 layout: packed routed experts,
    output-sharded host-local ``latent_up``, compact shared experts, split KDA
    gate projection, and paired attention output projections.
    """
    if not 2 <= layers <= 93:
        raise ValueError("Kimi stack supports a prefix of 2..93 layers")
    if vocab % 32:
        raise ValueError("Vocabulary must divide into 32 shards")
    m = max(1, layers // 4 + (layers == 93))
    k = layers - (layers // 4 + (layers == 93))
    shapes = dict(
        router=(layers - 1, hidden, 896),
        router_bias=(layers - 1, 1, 896),
        latent_down=(layers - 1, hidden, 128),
        latent_up=(layers - 1, 3584, 1024),
        latent_norm=(layers - 1, 1, 4096),
        shared_gu=(layers - 1, hidden, 384),
        shared_down=(layers - 1, 192, hidden),
        expert_gu=(layers - 1, 112, 448, 1536),
        expert_gus=(layers - 1, 112, 112, 1536),
        expert_down=(layers - 1, 112, 96, 3584),
        expert_ds=(layers - 1, 112, 24, 3584),
        k_gate=(k, hidden, 128),
        k_projection=(k, hidden, 1664),
        k_fb=(k, 128, 384),
        k_o=(k, 768, hidden // 2),
        k_conv=(k, 4, 3, 3, 128),
        k_a_log=(k, 3, 1),
        k_dt=(k, 3, 128),
        k_norm=(k, 8, 128),
        m_qa=(m, hidden, 1536),
        m_ka=(m, hidden, 640),
        m_qb=(m, 1536, 768),
        m_kb=(m, 512, 768),
        m_gate=(m, hidden, 384),
        m_o=(m, 768, hidden // 2),
        m_qnorm=(m, 1, 1536),
        m_knorm=(m, 1, 512),
        attn_res=(layers, 1, hidden),
        mlp_res=(layers, 1, hidden),
        attn_norm=(layers, 1, hidden),
        ffn_norm=(layers, 1, hidden),
        embedding=(vocab // 32, hidden),
        lm_head=(hidden, vocab // 32),
        dense_gu=(hidden, 2304),
        dense_down=(1152, hidden),
        output_res=(1, hidden),
        final_norm=(1, hidden),
    )
    return shapes


def _family_layers(config, name, layers):
    """Checkpoint layer ids stored along a family's leading axis (None: not per layer)."""
    kda = {"k_gate", "k_projection", "k_fb", "k_o", "k_conv", "k_a_log", "k_dt", "k_norm"}
    mla = {"m_qa", "m_ka", "m_qb", "m_kb", "m_gate", "m_o", "m_qnorm", "m_knorm"}
    moe = {
        "router", "router_bias", "latent_down", "latent_up", "latent_norm",
        "shared_gu", "shared_down", "expert_gu", "expert_gus", "expert_down", "expert_ds",
    }
    norms = {"attn_res", "mlp_res", "attn_norm", "ffn_norm"}
    if name in kda:
        return [i for i in range(layers) if i not in config.full_attention]
    if name in mla:
        return [i for i in range(layers) if i in config.full_attention] or [3]
    if name in moe:
        return list(range(1, layers))
    if name in norms:
        return list(range(layers))
    return None


def synthetic_rank_weights(rank, *, layers=93, hidden=7168, vocab=163840, seed=0, include_lm_head=False):
    """Deterministic random weights for one rank in the selected layout.

    The generated arrays have the same families, shapes, dtypes and padding
    structure as the device loader, with scales chosen so FP4 experts remain
    finite. They are used to compile against abstract weight signatures.
    """
    shapes = selected_local_shapes(layers, hidden, vocab=vocab)
    replicated = {
        "router", "router_bias", "latent_norm", "m_qa", "m_ka", "m_qnorm", "m_knorm", "k_norm",
        "attn_res", "mlp_res", "attn_norm", "ffn_norm", "output_res", "final_norm",
    }
    fp32 = {"router_bias", "latent_norm", "k_a_log", "k_dt", "attn_res", "mlp_res", "output_res"}
    fan_in = {"k_o": 12288, "m_o": 12288, "latent_up": 3584, "shared_down": 6144, "dense_down": 33792}
    for index, (name, shape) in enumerate(shapes.items()):
        if name == "lm_head" and not include_lm_head:
            continue
        rng = np.random.default_rng((seed, 300 + index, 0 if name in replicated else rank))
        if name in ("expert_gu", "expert_down"):
            value = rng.integers(0, 2**32, size=shape, dtype=np.uint64).astype(np.uint32)
        elif name in ("expert_gus", "expert_ds"):
            value = rng.integers(118, 121, size=shape, dtype=np.uint8)
        elif name == "latent_norm":
            value = np.broadcast_to((np.arange(4096) < 3584).astype(np.float32), shape).copy()
        else:
            value = rng.standard_normal(shape, dtype=np.float32)
            if name in ("attn_norm", "ffn_norm", "k_norm", "m_qnorm", "m_knorm", "final_norm"):
                value = 1 + value * 0.05
            elif name in ("attn_res", "mlp_res", "output_res"):
                value *= 0.1 / np.sqrt(hidden)
            elif name in ("router_bias", "k_a_log", "k_dt"):
                value *= 0.05
            elif (len(shape) >= 3 or name in ("dense_gu", "dense_down", "lm_head")) and name != "k_conv":
                value /= np.sqrt(fan_in.get(name, shape[-2]))
            else:
                value *= 0.15
            if name == "dense_gu":
                value[:, 1056:1152] = 0
                value[:, 2208:] = 0
            if name == "dense_down":
                value[1056:] = 0
            if name in ("latent_down",) and rank >= 28:
                value[...] = 0
            if name == "latent_up" and rank % 8 == 7:
                value[...] = 0
            if name == "m_ka":
                value[:, :, 576:] = 0
            if name == "m_qb":
                value = value.reshape(shape[0], 1536, 3, 256)
                value[:, :, :, 192:] = 0
                value = value.reshape(shape)
            if name not in fp32:
                value = value.astype(BF16)
        yield name, value


FP8_BLOCK = int(os.environ.get("K3_FP8_BLOCK", "256"))  # coarse FP8 scale block of the gate/up reblock


def _reblock_one(packed, scales, block_size):
    """Losslessly encode one MXFP4 matrix as FP8 plus coarse BF16 scales."""
    values = unpack_mxfp4(packed, scales)
    contraction_size, output_size = values.shape
    exponents = scales.reshape(
        contraction_size // block_size,
        block_size // 32,
        output_size,
    ).astype(jnp.int32)
    common_exponents = jnp.max(exponents, axis=1) - 6
    coarse_scales = jnp.exp2(common_exponents.astype(jnp.float32) - 127).astype(jnp.bfloat16)
    encoded = (
        values.reshape(contraction_size // block_size, block_size, output_size).astype(jnp.float32)
        / coarse_scales.astype(jnp.float32)[:, None, :]
    ).astype(jnp.float8_e4m3fn)
    restored = (
        encoded.astype(jnp.bfloat16) * coarse_scales[:, None, :]
    ).reshape(contraction_size, output_size)
    restored_bits = jax.lax.bitcast_convert_type(restored, jnp.uint16)
    value_bits = jax.lax.bitcast_convert_type(values, jnp.uint16)
    return encoded.reshape(values.shape), coarse_scales, jnp.all(restored_bits == value_bits)


def fp8_expert_shapes(layers: int = 93):
    """Per-rank shapes and dtypes of the FP8 gate/up storage (the loader's output)."""
    shapes = selected_local_shapes(layers, 7168)
    layer_count, experts, packed_rows, combined = shapes["expert_gu"]
    contraction = packed_rows * 8
    return {
        "expert_gu": ((layer_count, experts, contraction, combined), jnp.float8_e4m3fn),
        "expert_gus": ((layer_count, experts, contraction // FP8_BLOCK, combined), jnp.bfloat16),  # FP8_BLOCK: K3_FP8_BLOCK env
    }


def _make_layer_updates(device):
    """Per-device jitted updates: reblock one layer of experts to FP8 and store it,
    and store one layer of the packed down projection."""
    def update_gate_up(encoded, coarse, packed, scales, index):
        def convert(pair):
            return _reblock_one(pair[0], pair[1], FP8_BLOCK)

        layer_encoded, layer_coarse, valid = jax.lax.map(convert, (packed, scales), batch_size=1)
        encoded = jax.lax.dynamic_update_slice(encoded, layer_encoded[None, None], (0, index, 0, 0, 0))
        coarse = jax.lax.dynamic_update_slice(coarse, layer_coarse[None, None], (0, index, 0, 0, 0))
        return encoded, coarse, jnp.all(valid)

    def update_down(down, down_scales, packed, scales, index):
        down = jax.lax.dynamic_update_slice(down, packed[None, None], (0, index, 0, 0, 0))
        down_scales = jax.lax.dynamic_update_slice(down_scales, scales[None, None], (0, index, 0, 0, 0))
        return down, down_scales

    return (
        jax.jit(update_gate_up, donate_argnums=(0, 1)),
        jax.jit(update_down, donate_argnums=(0, 1)),
    )


def is_presharded(path) -> bool:
    """True for a checkpoint directory in the pre-sharded TP32 layout."""
    return (Path(path) / "layout.json").is_file()


def _read_presharded_dense(rank_dir: Path, index: dict, device, arrays: dict, emit, rank):
    """Sequentially stream ``dense.bin`` and upload each family."""
    families = [(name, spec) for name, spec in index["families"].items() if "file" in spec]
    families.sort(key=lambda item: item[1]["offset"])
    with open(rank_dir / "dense.bin", "rb", buffering=0) as handle:
        for name, spec in families:
            handle.seek(spec["offset"])
            count = int(np.prod(spec["shape"]))
            value = np.fromfile(handle, dtype=np.dtype(spec["dtype"]), count=count).reshape(spec["shape"])
            arrays[name] = jax.device_put(value[None], device)
            del value
            emit({"kind": "family", "rank": rank, "name": name})
    if arrays["k_projection"].shape[-1] == 1792:
        # Layouts written before the gate split carry the combined projection:
        # split it on the device into the gate columns and the rest.
        # (No donation: neither output has the input's shape, so the buffer
        # could not be reused and JAX would only warn.)
        combined = arrays["k_projection"]
        arrays["k_gate"], arrays["k_projection"] = jax.jit(
            lambda value: (
                value[..., 1152:1280],
                jnp.concatenate((value[..., :1152], value[..., 1280:]), axis=-1),
            )
        )(combined)
        del combined
        emit({"kind": "family", "rank": rank, "name": "k_gate"})


def _presharded_layer_reader(rank_dir: Path, spec: dict, layer_shape, dtype, queue, depth):
    """Read one expert family layer by layer from its files into ``queue`` (None at the end)."""
    count = int(np.prod(layer_shape))
    for filename, (lo, hi) in zip(spec["files"], spec["layer_ranges"]):
        with open(rank_dir / filename, "rb", buffering=0) as handle:
            for _ in range(lo, hi):
                queue.put(np.fromfile(handle, dtype=dtype, count=count).reshape(layer_shape))
    queue.put(None)


def _load_rank_presharded(path: Path, rank: int, device, layers: int, fp8_shapes, shapes, emit):
    """One rank from the pre-sharded layout: dense file in one stream, the four
    expert families in one stream each, gate/up converted to FP8 per layer."""
    import queue as queue_module
    import threading

    rank_dir = Path(path) / f"rank{rank:02d}"
    index = json.loads((rank_dir / "index.json").read_text())
    if index.get("format") != FORMAT_PRESHARDED or index["layers"] != layers:
        raise ValueError(f"{rank_dir}: unexpected layout format/layers {index.get('format')}/{index.get('layers')}")
    placement = SingleDeviceSharding(device)
    arrays = {}
    dense_thread = threading.Thread(
        target=_read_presharded_dense, args=(rank_dir, index, device, arrays, emit, rank), daemon=True
    )
    dense_thread.start()

    def zeros(shape, dtype):
        return jax.jit(lambda: jnp.zeros((1, *shape), dtype), out_shardings=placement)()

    encoded = zeros(*fp8_shapes["expert_gu"])
    coarse = zeros(*fp8_shapes["expert_gus"])
    down = zeros(shapes["expert_down"], jnp.uint32)
    down_scales = zeros(shapes["expert_ds"], jnp.uint8)
    update_gate_up, update_down = _make_layer_updates(device)
    queues = {}
    for name, dtype in (("expert_gu", np.uint32), ("expert_gus", np.uint8), ("expert_down", np.uint32), ("expert_ds", np.uint8)):
        spec = index["families"][name]
        queues[name] = queue_module.Queue(maxsize=2)
        threading.Thread(
            target=_presharded_layer_reader,
            args=(rank_dir, spec, tuple(spec["shape"][1:]), dtype, queues[name], 2), daemon=True,
        ).start()
    valid_flags = []
    for index_in_family in range(layers - 1):
        chunks = {name: queues[name].get() for name in queues}
        packed = jax.device_put(chunks["expert_gu"], device)
        packed_scales = jax.device_put(chunks["expert_gus"], device)
        encoded, coarse, valid = update_gate_up(encoded, coarse, packed, packed_scales, jnp.int32(index_in_family))
        valid_flags.append(valid)
        down, down_scales = update_down(
            down, down_scales, jax.device_put(chunks["expert_down"], device),
            jax.device_put(chunks["expert_ds"], device), jnp.int32(index_in_family),
        )
        del chunks, packed, packed_scales
        emit({"kind": "layer", "rank": rank, "layer": index_in_family + 1, "layers": layers - 1})
    for name in queues:
        if queues[name].get() is not None:
            raise ValueError(f"{rank_dir}: {name} has more layers than expected")
    dense_thread.join()
    arrays["expert_gu"] = encoded
    arrays["expert_gus"] = coarse
    arrays["expert_down"] = down
    arrays["expert_ds"] = down_scales
    jax.block_until_ready(arrays)
    if not all(bool(flag) for flag in valid_flags):
        raise ValueError(f"rank {rank}: FP8 reblocking did not reconstruct the gate/up weights exactly")
    emit({"kind": "rank", "rank": rank})
    return arrays


FORMAT_PRESHARDED = 1


def load_weights(path, mesh, *, layers: int = 93, ranks_in_flight: int = 8,
                 progress: Callable[[dict], None] | None = None, log=None):
    """Load the checkpoint (HF layout or pre-sharded TP32 layout) into
    ``[32, ...]`` device arrays with FP8 gate/up experts."""
    if mesh.size != 32:
        raise ValueError("K3 requires 32 devices")
    presharded = is_presharded(path)
    if presharded:
        layout = json.loads((Path(path) / "layout.json").read_text())
        if layout["layers"] != layers:
            raise ValueError(f"pre-sharded layout has {layout['layers']} layers, requested {layers}")
        shapes = selected_local_shapes(layers, 7168)
        fp8_shapes = fp8_expert_shapes(layers)
        sharding = NamedSharding(mesh, P("tp"))
        devices = list(mesh.devices.flat)
        local_ranks = [rank for rank, device in enumerate(devices) if device.process_index == jax.process_index()]

        def emit(event):
            if progress is not None:
                progress(event)

        started = time.perf_counter()
        per_rank = {}
        with ThreadPoolExecutor(max(1, ranks_in_flight)) as pool:
            futures = {
                rank: pool.submit(_load_rank_presharded, Path(path), rank, devices[rank], layers, fp8_shapes, shapes, emit)
                for rank in local_ranks
            }
            for rank, future in futures.items():
                per_rank[rank] = future.result()
                if log is not None:
                    log(f"rank {rank} resident after {time.perf_counter() - started:.0f} s")
        weights = {}
        for name, shape in shapes.items():
            if name in fp8_shapes:
                shape = fp8_shapes[name][0]
            weights[name] = jax.make_array_from_single_device_arrays(
                (32, *shape), sharding, [per_rank[rank][name] for rank in local_ranks]
            )
        return weights
    checkpoint = Checkpoint(path)
    shapes = selected_local_shapes(layers, checkpoint.config.dim, vocab=checkpoint.config.vocab)
    fp8_shapes = fp8_expert_shapes(layers)
    sharding = NamedSharding(mesh, P("tp"))
    devices = list(mesh.devices.flat)
    local_ranks = [rank for rank, device in enumerate(devices) if device.process_index == jax.process_index()]
    experts_per_rank = checkpoint.config.experts // 8
    expert_layers = layers - 1

    def emit(event):
        if progress is not None:
            progress(event)

    def load_rank(rank):
        device = devices[rank]
        placement = SingleDeviceSharding(device)
        arrays = {}
        # Dense families (everything except the four expert families).
        for name, shape in shapes.items():
            if name in ("expert_gu", "expert_gus", "expert_down", "expert_ds"):
                continue
            ids = _family_layers(checkpoint.config, name, layers)
            if ids is None:
                value = checkpoint.dense_selected(name, 0, rank)
            else:
                first = checkpoint.dense_selected(name, ids[0], rank)
                value = np.empty(shape, first.dtype)
                value[0] = first
                with ThreadPoolExecutor(checkpoint.READ_THREADS) as pool:
                    for i, layer_value in zip(
                        range(1, len(ids)),
                        pool.map(lambda layer: checkpoint.dense_selected(name, layer, rank), ids[1:]),
                    ):
                        value[i] = layer_value
            if value.shape != shape:
                raise ValueError((name, value.shape, shape))
            arrays[name] = jax.device_put(value[None], device)
            del value
            emit({"kind": "family", "rank": rank, "name": name})

        # Expert families, one layer at a time, converted on the device.
        def zeros(shape, dtype):
            return jax.jit(lambda: jnp.zeros((1, *shape), dtype), out_shardings=placement)()

        encoded = zeros(*fp8_shapes["expert_gu"])
        coarse = zeros(*fp8_shapes["expert_gus"])
        down = zeros(shapes["expert_down"], jnp.uint32)
        down_scales = zeros(shapes["expert_ds"], jnp.uint8)
        update_gate_up, update_down = _make_layer_updates(device)
        first_expert = experts_per_rank * (rank // 4)
        part = rank % 4
        valid_flags = []
        for index, layer in enumerate(range(1, layers)):
            gate_up, gate_up_scales, downs, down_scale = checkpoint.experts(
                layer, first_expert, experts_per_rank, part=part, parts=4
            )
            packed = jax.device_put(np.asarray(gate_up, np.uint32), device)
            packed_scales = jax.device_put(np.asarray(gate_up_scales, np.uint8), device)
            encoded, coarse, valid = update_gate_up(encoded, coarse, packed, packed_scales, jnp.int32(index))
            valid_flags.append(valid)
            down, down_scales = update_down(
                down, down_scales,
                jax.device_put(np.asarray(downs, np.uint32), device),
                jax.device_put(np.asarray(down_scale, np.uint8), device),
                jnp.int32(index),
            )
            del gate_up, gate_up_scales, downs, down_scale, packed, packed_scales
            emit({"kind": "layer", "rank": rank, "layer": layer, "layers": expert_layers})
        arrays["expert_gu"] = encoded
        arrays["expert_gus"] = coarse
        arrays["expert_down"] = down
        arrays["expert_ds"] = down_scales
        jax.block_until_ready(arrays)
        if not all(bool(flag) for flag in valid_flags):
            raise ValueError(f"rank {rank}: FP8 reblocking did not reconstruct the gate/up weights exactly")
        emit({"kind": "rank", "rank": rank})
        return rank, arrays

    started = time.perf_counter()
    per_rank = {}
    with ThreadPoolExecutor(max(1, ranks_in_flight)) as pool:
        for rank, arrays in pool.map(load_rank, local_ranks):
            per_rank[rank] = arrays
            if log is not None:
                log(f"rank {rank} resident after {time.perf_counter() - started:.0f} s")
    weights = {}
    for name, shape in shapes.items():
        if name in fp8_shapes:
            shape = fp8_shapes[name][0]
        weights[name] = jax.make_array_from_single_device_arrays(
            (32, *shape), sharding, [per_rank[rank][name] for rank in local_ranks]
        )
    return weights


def abstract_weights(mesh, path=None, *, layers: int = 93, vocab: int = 163840):
    """``jax.ShapeDtypeStruct`` tree matching :func:`load_weights`'s result, for
    compiling programs before the weights are resident.

    With ``path`` the dense families' dtypes are read from the checkpoint (one
    layer of each family for rank 0; ``k_norm`` and ``k_conv`` are FP32 there
    while the synthetic fixture uses BF16). Without it the fixture's dtypes
    are used.
    """
    shapes = selected_local_shapes(layers, 7168, vocab=vocab)
    dtypes = {name: value.dtype for name, value in synthetic_rank_weights(0, layers=2, include_lm_head=True)}
    if path is not None and is_presharded(path):
        index = json.loads((Path(path) / "rank00" / "index.json").read_text())
        for name, spec in index["families"].items():
            dtypes[name] = np.dtype(spec["dtype"])
    elif path is not None:
        checkpoint = Checkpoint(path)
        for name in shapes:
            if name in ("expert_gu", "expert_gus", "expert_down", "expert_ds"):
                continue
            ids = _family_layers(checkpoint.config, name, layers)
            dtypes[name] = checkpoint.dense_selected(name, 0 if ids is None else ids[0], 0).dtype
    dtypes["k_gate"] = dtypes["k_projection"]  # split from it when the layout predates k_gate
    fp8_shapes = fp8_expert_shapes(layers)
    sharding = NamedSharding(mesh, P("tp"))
    out = {}
    for name, shape in shapes.items():
        dtype = dtypes[name]
        if name in fp8_shapes:
            shape, dtype = fp8_shapes[name]
        out[name] = jax.ShapeDtypeStruct((32, *shape), dtype, sharding=sharding)
    return out
