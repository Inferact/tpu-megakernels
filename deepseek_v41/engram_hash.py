"""Host-side int64 Engram n-gram hashing for DeepSeek-V4.1-Flash.

Matches `REF/engram.py:9-185` and `INVENTORY.md` §4.1:
- `find_next_prime`: smallest unused prime above `start` (`15,999,999` in full config, or
  `cfg.engram_vocab_size - 1` in `tiny_config`), drawn in order `layer -> n-gram -> head`.
- `build_compressed_token_map`: normalizes every token via the exact `tokenizers` NFKC -> NFD ->
  StripAccents -> Lowercase -> whitespace collapse -> single-space sentinel -> Strip sequence
  (`REF/engram.py:28-61`), with a deterministic fallback (`id % compressed_vocab_size`) and
  `SyntheticTokenizer` when running without `tokenizer.json`.
- `compute_hash_multipliers`: per-layer odd `int64` multipliers from `np.random.default_rng(10007 * layer_id)`.
- `EngramHashState` / `compute_engram_hash_ids`: computes the 24 row IDs per token per Engram layer
  with `pad = token_map[engram_pad_id]` propagation whenever `p < shift` or across dead image tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sympy import isprime

from deepseek_v41.config import DSV41Config

DEFAULT_COMPRESSED_VOCAB_SIZE = 99092


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """Returns the smallest prime strictly above `start` not in `seen_primes` (`REF/engram.py:9-14`)."""
    candidate = int(start) + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


class _SyntheticBackendTokenizer:
    """Minimal backend tokenizer compatible with `REF/engram.py:build_compressed_token_map`."""

    def __init__(self, vocab_size: int, compressed_vocab_size: int):
        self.vocab_size = int(vocab_size)
        self.compressed_vocab_size = int(compressed_vocab_size)

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        del with_added_tokens
        return self.vocab_size

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        tid = int(ids[0])
        return f"tok_{tid % self.compressed_vocab_size}"

    def id_to_token(self, token_id: int) -> str:
        return f"tok_{int(token_id) % self.compressed_vocab_size}"


class SyntheticTokenizer:
    """Deterministic synthetic tokenizer for unit tests and `tiny_config` runs without `tokenizer.json`.

    Both `REF/engram.py:build_compressed_token_map(SyntheticTokenizer(V, C))` and our fallback
    produce `token_map[i] = i % C` and `compressed_vocab_size = C`.
    """

    def __init__(self, vocab_size: int = 1024, compressed_vocab_size: int | None = None):
        self.vocab_size = int(vocab_size)
        self.compressed_vocab_size = int(compressed_vocab_size or vocab_size)
        self.backend_tokenizer = _SyntheticBackendTokenizer(self.vocab_size, self.compressed_vocab_size)
        self.eos_token_id = 1

    def __len__(self) -> int:
        return self.vocab_size


def build_compressed_token_map(
    tokenizer_or_path: Any = None,
    *,
    vocab_size: int = 129280,
    compressed_vocab_size: int | None = None,
) -> tuple[np.ndarray, int]:
    """Builds the compressed token lookup table matching `REF/engram.py:17-61`.

    Args:
        tokenizer_or_path: A HuggingFace tokenizer (with `.backend_tokenizer`), a `tokenizers.Tokenizer`,
            a path to `tokenizer.json` (or directory containing it), a `SyntheticTokenizer`, or `None`.
        vocab_size: Fallback vocabulary size when `tokenizer_or_path is None`.
        compressed_vocab_size: Target compressed vocabulary size when `tokenizer_or_path is None`
            (defaults to `DEFAULT_COMPRESSED_VOCAB_SIZE` if `vocab_size == 129280` else `vocab_size`).

    Returns:
        `(token_map_int64, compressed_vocab_size)` where `token_map_int64` has shape `[vocab_size]`.
    """
    if tokenizer_or_path is None:
        target_c = compressed_vocab_size or (
            DEFAULT_COMPRESSED_VOCAB_SIZE if vocab_size == 129280 else vocab_size
        )
        lookup = np.arange(vocab_size, dtype=np.int64) % int(target_c)
        return lookup, int(target_c)

    from tokenizers import Regex, Tokenizer, normalizers

    if isinstance(tokenizer_or_path, (str, Path)):
        p = Path(tokenizer_or_path)
        if p.is_dir():
            p = p / "tokenizer.json"
        backend = Tokenizer.from_file(str(p))
        total_vocab = backend.get_vocab_size(with_added_tokens=True)
    elif hasattr(tokenizer_or_path, "backend_tokenizer"):
        backend = tokenizer_or_path.backend_tokenizer
        total_vocab = len(tokenizer_or_path)
    elif isinstance(tokenizer_or_path, Tokenizer):
        backend = tokenizer_or_path
        total_vocab = backend.get_vocab_size(with_added_tokens=True)
    else:
        raise TypeError(f"Unsupported tokenizer_or_path: {type(tokenizer_or_path)!r}")

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    key_to_new: dict[str, int] = {}
    lookup = np.zeros(total_vocab, dtype=np.int64)
    for token_id in range(total_vocab):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...],
    max_ngram_size: int = 4,
    tokenizer_vocab_size: int = DEFAULT_COMPRESSED_VOCAB_SIZE,
) -> np.ndarray:
    """Computes per-layer odd `int64` n-gram multipliers matching `REF/engram.py:64-83`."""
    max_long = int(np.iinfo(np.int64).max)
    multiplier_bound = max(1, (max_long // int(tokenizer_vocab_size)) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * int(layer_id))
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(values * np.int64(2) + np.int64(1))
    if not rows:
        return np.zeros((0, max_ngram_size), dtype=np.int64)
    return np.stack(rows, axis=0)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout and prime moduli for the Engram hash tables (`REF/engram.py:86-127`)."""

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    offsets: tuple[tuple[int, ...], ...]
    n_heads: int
    head_dim: int

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @property
    def primes_np(self) -> np.ndarray:
        return np.asarray(self.primes, dtype=np.int64)

    @property
    def offsets_np(self) -> np.ndarray:
        return np.asarray(self.offsets, dtype=np.int64)

    @classmethod
    def from_config(cls, cfg: DSV41Config | Any) -> EngramLayout | None:
        layer_ids = tuple(cfg.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size = int(cfg.engram_max_ngram_size)
        n_heads = int(cfg.engram_n_heads)
        vocab_start = int(cfg.engram_vocab_size)
        primes: list[tuple[tuple[int, ...], ...]] = []
        offsets: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for _ in layer_ids:
            per_ngram: list[tuple[int, ...]] = []
            flat_layer: list[int] = []
            for _ in range(max_ngram_size - 1):
                sizes: list[int] = []
                current = vocab_start - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
                flat_layer.extend(sizes)
            primes.append(tuple(per_ngram))
            layer_offsets = np.cumsum([0, *flat_layer[:-1]], dtype=np.int64).tolist()
            offsets.append(tuple(int(x) for x in layer_offsets))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(int(x) for x in cfg.engram_num_embeddings),
            primes=tuple(primes),
            offsets=tuple(offsets),
            n_heads=n_heads,
            head_dim=int(cfg.engram_head_dim),
        )

    @classmethod
    def from_args(cls, args: Any) -> EngramLayout | None:
        return cls.from_config(args)


class EngramHashState:
    """Stateful host-side `int64` n-gram hasher matching `REF/engram.py:129-185`."""

    DEAD = -1

    def __init__(
        self,
        cfg: DSV41Config | Any,
        layout: EngramLayout | None = None,
        tokenizer: Any = None,
        *,
        max_batch_size: int = 32,
        max_seq_len: int = 65536,
        compressed_vocab_size: int | None = None,
    ):
        self.layout = layout if layout is not None else EngramLayout.from_config(cfg)
        if self.layout is None:
            raise ValueError("Engram is disabled on this config (`engram_layer_ids` is empty)")
        default_c = getattr(cfg, "engram_compressed_vocab_size", None) or compressed_vocab_size
        if default_c is None:
            default_c = DEFAULT_COMPRESSED_VOCAB_SIZE if cfg.vocab_size == 129280 else cfg.vocab_size
        token_map, actual_c = build_compressed_token_map(
            tokenizer,
            vocab_size=cfg.vocab_size,
            compressed_vocab_size=default_c,
        )
        self.compressed_vocab_size = actual_c
        self.token_map = np.asarray(token_map, dtype=np.int64)
        self.pad_id = int(self.token_map[int(cfg.engram_pad_id)])
        self.primes = self.layout.primes_np
        self.offsets = self.layout.offsets_np
        self.multipliers = compute_hash_multipliers(
            self.layout.layer_ids,
            self.layout.max_ngram_size,
            self.compressed_vocab_size,
        )
        bsz = int(getattr(cfg, "max_batch_size", max_batch_size))
        seq_len = int(getattr(cfg, "max_seq_len", max_seq_len))
        self.cache = np.full((bsz, seq_len), self.pad_id, dtype=np.int64)

    def reset(self) -> None:
        self.cache.fill(self.pad_id)

    def forward(
        self,
        input_ids: np.ndarray,
        start_pos: int = 0,
        token_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Computes `int64` Engram row IDs of shape `[B, L, n_engram_layers, 24]` (`REF/engram.py:160-185`)."""
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        batch, seqlen = ids.shape
        if batch > self.cache.shape[0] or start_pos + seqlen > self.cache.shape[1]:
            new_b = max(batch, self.cache.shape[0])
            new_s = max(start_pos + seqlen, self.cache.shape[1])
            grown = np.full((new_b, new_s), self.pad_id, dtype=np.int64)
            grown[: self.cache.shape[0], : self.cache.shape[1]] = self.cache
            self.cache = grown

        compressed = self.token_map[ids]
        if token_mask is not None:
            mask_bool = np.asarray(token_mask, dtype=bool)
            compressed = np.where(mask_bool, compressed, np.int64(self.DEAD))
        self.cache[:batch, start_pos : start_pos + seqlen] = compressed

        positions = np.broadcast_to(
            np.arange(start_pos, start_pos + seqlen, dtype=np.int64)[None, :],
            (batch, seqlen),
        )
        tokens = []
        blocked = np.zeros((batch, seqlen), dtype=bool)
        batch_idx = np.arange(batch, dtype=np.int64)[:, None]
        for shift in range(self.layout.max_ngram_size):
            gather_pos = np.maximum(positions - shift, 0)
            source = self.cache[batch_idx, gather_pos]
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(np.where(blocked, np.int64(self.pad_id), source))
        tokens_arr = np.stack(tokens, axis=-1)  # [B, L, max_ngram_size]

        # [B, L, 1, max_ngram_size] * [1, 1, n_engram_layers, max_ngram_size]
        products = tokens_arr[:, :, None, :] * self.multipliers[None, None, :, :]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, self.layout.max_ngram_size):
            rolling = np.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % self.primes[None, None, :, i - 1, :])
        raw_ids = np.concatenate(hashes, axis=-1) + self.offsets[None, None, :, :]
        return raw_ids

    def __call__(
        self,
        input_ids: np.ndarray,
        start_pos: int = 0,
        token_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.forward(input_ids, start_pos=start_pos, token_mask=token_mask)


NgramHashState = EngramHashState


def compute_engram_hash_ids(
    input_ids: np.ndarray,
    cfg: DSV41Config,
    *,
    tokenizer: Any = None,
    compressed_vocab_size: int | None = None,
    token_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Stateless helper that hashes a full prefix `input_ids` (`[B, L]`) from `start_pos=0`."""
    ids = np.asarray(input_ids, dtype=np.int64)
    if ids.ndim == 1:
        ids = ids[None, :]
    bsz, seqlen = ids.shape
    state = EngramHashState(
        cfg,
        tokenizer=tokenizer,
        max_batch_size=bsz,
        max_seq_len=max(seqlen, 16),
        compressed_vocab_size=compressed_vocab_size,
    )
    return state.forward(ids, start_pos=0, token_mask=token_mask)
