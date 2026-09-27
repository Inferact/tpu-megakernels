"""Unified DeepSeek-V4.1-Flash inference engine (`DSV41Engine`) for TPU v7x and CPU meshes.

Wraps:
- Chunked prefill (`PREFILL_CHUNK_SIZE = 8` + single-token tail so only 2 sequence lengths compile)
- Single-token decode and `q = 1 + k` block verification via either:
  - `decode_megakernel.make_megakernel_decode` (Pallas TPU v7x megakernel)
  - `xla_decode.make_xla_decode_step` (same-run XLA-jitted control)
- 3-layer DSpark speculative drafting + greedy block verification (`dspark.DSparkDrafter`, `verify_and_commit_block`)
- Multi-host SPMD command broadcast (`jax.experimental.multihost_utils`) so coordinator process 0
  can drive HTTP server requests and evaluation loops while worker processes `1..P-1` execute in lockstep.
"""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from deepseek_v41.config import DSV41Config
from deepseek_v41.decode_megakernel import (
    compute_vmem_budget,
    make_megakernel_decode,
    make_tpu_pallas_megakernel_step,
    prepare_tpu_megakernel_weights,
)
from deepseek_v41.dspark import AcceptanceStats, DSparkDrafter, verify_and_commit_block
from deepseek_v41.xla_decode import init_cache, make_xla_decode_step


PREFILL_CHUNK_SIZE = 6
_STOP_ANSWER_RE = re.compile(
    r"(?:\nQuestion:|####\s*\-?\d[\d,\.]*\s*\n|(?:The\s+(?:correct\s+)?answer\s+is|Therefore,\s+the\s+(?:correct\s+)?answer\s+is)\s*\**[(\[]?[A-D][)\]]?\**\.?\s*\n)"
)

# Multi-host SPMD command IDs
CMD_NOP = 0
CMD_STOP = 1
CMD_GENERATE = 2
CMD_TEACHER_FORCED = 3
CMD_BENCHMARK = 4


class SimpleTokenizer:
    """Wrapper around HuggingFace `tokenizers.Tokenizer` (`tokenizer.json`) with deterministic fallback."""

    def __init__(self, tokenizer_path: str | Path | None = None, vocab_size: int = 129280) -> None:
        self.vocab_size = int(vocab_size)
        self.eos_token_id = 1
        self.bos_token_id = 0
        self._hf_tok = None
        if tokenizer_path is None and self.vocab_size == 129280:
            for cand in (
                Path(os.environ.get("TOKENIZER_PATH", "")) if os.environ.get("TOKENIZER_PATH") else None,
                Path(__file__).resolve().parent / "tokenizer.json",
            ):
                if cand is not None and cand.is_file():
                    tokenizer_path = cand
                    break
        if tokenizer_path is not None:
            p = Path(tokenizer_path)
            if p.is_dir():
                p = p / "tokenizer.json"
            if p.exists():
                from tokenizers import Tokenizer

                self._hf_tok = Tokenizer.from_file(str(p))
                eos_id = self._hf_tok.token_to_id("<｜end▁of▁sentence｜>")
                if eos_id is not None:
                    self.eos_token_id = int(eos_id)
                bos_id = self._hf_tok.token_to_id("<｜begin▁of▁sentence｜>")
                if bos_id is not None:
                    self.bos_token_id = int(bos_id)

    def encode(self, text: str, *, add_bos: bool = True) -> list[int]:
        if self._hf_tok is not None:
            ids = list(self._hf_tok.encode(text, add_special_tokens=False).ids)
            if add_bos and (not ids or ids[0] != self.bos_token_id):
                ids = [self.bos_token_id] + ids
            return ids
        raw = [int(b) + 3 for b in text.encode("utf-8")]
        if add_bos:
            raw = [self.bos_token_id] + raw
        return [t % self.vocab_size for t in raw]

    def decode(self, token_ids: list[int] | np.ndarray) -> str:
        ids = [int(t) for t in token_ids if int(t) not in (self.bos_token_id, self.eos_token_id)]
        if self._hf_tok is not None:
            return self._hf_tok.decode(ids, skip_special_tokens=True)
        byte_vals = [max(32, min(126, (t - 3) % 128)) for t in ids]
        return bytes(byte_vals).decode("ascii", errors="replace")

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        thinking_mode: str = "chat",
        reasoning_effort: int | str = 50,
    ) -> str:
        parts: list[str] = []
        if thinking_mode == "thinking":
            effort_map = {"low": 50, "high": 75, "max": 100}
            budget = effort_map.get(str(reasoning_effort).lower(), reasoning_effort)
            parts.append(
                f"<｜System｜>Reasoning Effort: {budget} "
                "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
            )
        for idx, m in enumerate(messages):
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                if idx == 0 and thinking_mode == "thinking":
                    parts.append(content)
                else:
                    parts.append(f"<｜System｜>{content}")
            elif role == "user":
                parts.append(f"<｜User｜>{content}")
            elif role == "assistant":
                parts.append(f"<｜Assistant｜></think>{content}<｜end▁of▁sentence｜>")
        if thinking_mode == "thinking":
            parts.append("<｜Assistant｜><think>")
        else:
            parts.append("<｜Assistant｜></think>")
        return "".join(parts)


class DSV41Engine:
    """Unified DeepSeek-V4.1-Flash inference & verification engine."""

    def __init__(
        self,
        cfg: DSV41Config,
        mesh: Mesh,
        sharded_weights: Mapping[str, jax.Array],
        *,
        tokenizer_path: str | Path | None = None,
        default_backend: str = "megakernel",
        sinkhorn_iters: int | None = None,
        index_k_mode: str | None = None,
        interpret: bool | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.mesh = mesh
        self._weights_holder = sharded_weights
        self.sharded_weights = dict(sharded_weights)
        self.default_backend = default_backend
        self.sinkhorn_iters = cfg.hc_sinkhorn_iters if sinkhorn_iters is None else int(sinkhorn_iters)
        if index_k_mode is None:
            self.index_k_mode = "intended" if cfg.n_layers >= 40 else cfg.index_k_mode
        else:
            self.index_k_mode = index_k_mode
        self.interpret = interpret
        self.max_seq_len = int(max_seq_len) if max_seq_len is not None else (256 if cfg.n_layers >= 40 else 64)
        self.tokenizer = SimpleTokenizer(tokenizer_path, vocab_size=cfg.vocab_size)
        if "engram.token_map" not in self.sharded_weights and tokenizer_path is not None:
            from deepseek_v41.engram_hash import build_compressed_token_map
            from deepseek_v41.xla_decode import replicate_to_mesh

            tmap_np, _ = build_compressed_token_map(
                tokenizer_path,
                vocab_size=cfg.vocab_size,
                compressed_vocab_size=getattr(
                    cfg, "engram_compressed_vocab_size", 99092 if cfg.vocab_size == 129280 else cfg.vocab_size
                ),
            )
            self.sharded_weights["engram.token_map"] = replicate_to_mesh(
                jnp.asarray(tmap_np, dtype=jnp.int32), self.mesh
            )

        self._step_fns: dict[tuple[str, int, str], Callable] = {}
        self._tpu_packed_weights: tuple[jax.Array, ...] | None = None
        self._tpu_pallas_fns: dict[int, Callable] = {}
        self._prefix_kv_cache: dict[tuple[Any, ...], tuple[int, dict[str, jax.Array], jax.Array, jax.Array]] = {}
        self.drafter = DSparkDrafter(cfg, mesh, sinkhorn_iters=self.sinkhorn_iters)
        self.vmem_budget = compute_vmem_budget(cfg, batch_size=8, num_devices=mesh.size, max_verify_tokens=6)

    def get_step_fn(
        self,
        backend: str | None = None,
        *,
        sinkhorn_iters: int | None = None,
        index_k_mode: str | None = None,
    ) -> Callable:
        b_name = (backend or self.default_backend).lower()
        s_iters = self.sinkhorn_iters if sinkhorn_iters is None else int(sinkhorn_iters)
        ik_mode = self.index_k_mode if index_k_mode is None else index_k_mode
        key = (b_name, s_iters, ik_mode)
        if key not in self._step_fns:
            if b_name == "megakernel":
                self._step_fns[key] = make_megakernel_decode(
                    self.cfg,
                    self.mesh,
                    sinkhorn_iters=s_iters,
                    index_k_mode=ik_mode,
                    interpret=self.interpret,
                )
            elif b_name == "xla":
                self._step_fns[key] = make_xla_decode_step(
                    self.cfg,
                    self.mesh,
                    sinkhorn_iters=s_iters,
                    index_k_mode=ik_mode,
                )
            else:
                raise ValueError(f"Unknown backend {b_name!r}; expected 'megakernel' or 'xla'")
        return self._step_fns[key]

    def init_cache(self, batch_size: int = 1) -> dict[str, jax.Array]:
        return init_cache(self.cfg, batch_size=batch_size, max_seq_len=self.max_seq_len, mesh=self.mesh)

    def prefill(
        self,
        prompt_ids: np.ndarray | jax.Array,
        *,
        cache: dict[str, jax.Array] | None = None,
        backend: str | None = None,
        sinkhorn_iters: int | None = None,
        update_dspark: bool = True,
        chunk_size: int = PREFILL_CHUNK_SIZE,
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array], float]:
        """Run chunked prefill over `prompt_ids` (`[B, L]`) and return `(last_logits, last_main_hidden, cache, elapsed_s)`."""
        ids = np.asarray(prompt_ids, dtype=np.int32)
        if ids.ndim == 1:
            ids = ids[None, :]
        bsz, seqlen = ids.shape
        can_memoize = cache is None and bsz == 1 and seqlen > 94 * chunk_size
        b_name = (backend or self.default_backend).lower()
        s_iters = self.sinkhorn_iters if sinkhorn_iters is None else int(sinkhorn_iters)
        snap_pos = 94 * chunk_size

        step_fn = self.get_step_fn(backend=backend, sinkhorn_iters=sinkhorn_iters)
        t0 = time.perf_counter()
        pos = 0
        last_logits = None
        last_mh = None

        if can_memoize:
            prefix_key = (b_name, s_iters, bool(update_dspark), tuple(ids[0, :snap_pos].tolist()))
            if prefix_key in self._prefix_kv_cache:
                pos, saved_cache, last_logits, last_mh = self._prefix_kv_cache[prefix_key]
                cache = dict(saved_cache)
            else:
                cache = self.init_cache(batch_size=bsz)
        elif cache is None:
            cache = self.init_cache(batch_size=bsz)

        # Process full chunks of `chunk_size` (default 6, matching 1 + dspark_block_size)
        while pos + chunk_size < seqlen:
            chunk = jnp.asarray(ids[:, pos : pos + chunk_size], dtype=jnp.int32)
            logits_c, mh_c, cache = step_fn(
                self.sharded_weights,
                cache,
                chunk,
                jnp.asarray(pos, dtype=jnp.int32),
            )
            if update_dspark and self.cfg.n_mtp_layers > 0:
                cache = self.drafter.prefill_main_hidden(self.sharded_weights, cache, mh_c, pos)
            last_logits = logits_c[:, -1, :]
            last_mh = mh_c[:, -1:, :]
            pos += chunk_size
            if can_memoize and pos == snap_pos:
                prefix_key = (b_name, s_iters, bool(update_dspark), tuple(ids[0, :snap_pos].tolist()))
                if len(self._prefix_kv_cache) >= 2:
                    self._prefix_kv_cache.clear()
                self._prefix_kv_cache[prefix_key] = (pos, dict(cache), last_logits, last_mh)

        # Process remaining tail tokens 1 by 1 so only chunk_size and 1-token shapes compile
        while pos < seqlen:
            tok = jnp.asarray(ids[:, pos : pos + 1], dtype=jnp.int32)
            logits_1, mh_1, cache = step_fn(
                self.sharded_weights,
                cache,
                tok,
                jnp.asarray(pos, dtype=jnp.int32),
            )
            if update_dspark and self.cfg.n_mtp_layers > 0 and pos + 1 < seqlen:
                cache = self.drafter.prefill_main_hidden(self.sharded_weights, cache, mh_1, pos)
            last_logits = logits_1[:, -1, :] if logits_1.ndim == 3 else logits_1
            last_mh = mh_1[:, None, :] if mh_1.ndim == 2 else mh_1
            pos += 1

        assert last_logits is not None and last_mh is not None
        jax.block_until_ready(last_logits)
        elapsed_s = time.perf_counter() - t0
        return last_logits, last_mh, cache, elapsed_s

    def teacher_forced_logits(
        self,
        token_ids: np.ndarray | jax.Array,
        *,
        backend: str | None = None,
        sinkhorn_iters: int | None = None,
    ) -> jax.Array:
        """Compute teacher-forced `[B, L, vocab_size]` logits across `token_ids` (`[B, L]`)."""
        ids = np.asarray(token_ids, dtype=np.int32)
        if ids.ndim == 1:
            ids = ids[None, :]
        bsz, seqlen = ids.shape
        cache = self.init_cache(batch_size=bsz)
        step_fn = self.get_step_fn(backend=backend, sinkhorn_iters=sinkhorn_iters)

        logits_list: list[jax.Array] = []
        for pos in range(seqlen):
            tok = jnp.asarray(ids[:, pos : pos + 1], dtype=jnp.int32)
            logits_1, _, cache = step_fn(
                self.sharded_weights,
                cache,
                tok,
                jnp.asarray(pos, dtype=jnp.int32),
            )
            logits_list.append(logits_1[:, 0, :] if logits_1.ndim == 3 else logits_1)
        all_logits = jnp.stack(logits_list, axis=1)
        jax.block_until_ready(all_logits)
        return all_logits

    def generate_greedy(
        self,
        prompt_ids: list[int] | np.ndarray,
        max_new_tokens: int,
        *,
        use_dspark: bool = False,
        backend: str | None = None,
        eos_token_id: int | None = None,
        confidence_threshold: float | None = None,
        force_accept_schedule: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run greedy autoregressive or DSpark speculative decoding for `prompt_ids` (`B=1`)."""
        ids_np = np.asarray(prompt_ids, dtype=np.int32)
        if ids_np.ndim == 1:
            ids_np = ids_np[None, :]
        if ids_np.shape[0] != 1:
            raise ValueError(f"generate_greedy requires batch_size == 1, got {ids_np.shape[0]}")
        max_prompt_allowed = max(16, self.max_seq_len - 64)
        if ids_np.shape[1] > max_prompt_allowed:
            ids_np = ids_np[:, -max_prompt_allowed:]
        prompt_len = int(ids_np.shape[1])
        max_new_tokens = min(int(max_new_tokens), max(0, self.max_seq_len - prompt_len - 8))
        if max_new_tokens <= 0:
            return {
                "prompt_ids": ids_np[0].tolist(),
                "generated_ids": [],
                "full_ids": ids_np[0].tolist(),
                "prefill_s": 0.0,
                "decode_s": 0.0,
                "tok_per_s": 0.0,
                "step_times_ms": [],
                "median_step_ms": 0.0,
                "p99_step_ms": 0.0,
                "dspark_stats": AcceptanceStats(self.cfg.dspark_block_size).to_dict(),
            }

        step_fn = self.get_step_fn(backend=backend)
        last_logits, last_mh, cache, prefill_s = self.prefill(
            ids_np,
            backend=backend,
            update_dspark=use_dspark,
        )
        if eos_token_id is not None:
            first_logits = np.asarray(last_logits, dtype=np.float32).copy()
            if 0 <= int(eos_token_id) < first_logits.shape[-1]:
                first_logits[..., int(eos_token_id)] = -1e9
            first_tok = int(np.argmax(first_logits, axis=-1).ravel()[0])
        else:
            first_tok = int(np.asarray(jnp.argmax(last_logits, axis=-1)).ravel()[0])
        generated: list[int] = [first_tok]
        step_times_ms: list[float] = []
        dspark_stats = AcceptanceStats(self.cfg.dspark_block_size)

        # Commit position of the last prompt token whose main_hidden is `last_mh`
        cur_pos = prompt_len
        pending_mh = last_mh  # [1, 1, 4*dim] from position `prompt_len - 1`
        pending_mh_start_pos = prompt_len - 1
        cur_tok = first_tok
        step_idx = 0

        t_decode_start = time.perf_counter()
        while len(generated) < max_new_tokens:
            t_step0 = time.perf_counter()
            if not use_dspark or self.cfg.n_mtp_layers <= 0 or self.cfg.dspark_block_size <= 0:
                tok_arr = jnp.asarray([[cur_tok]], dtype=jnp.int32)
                logits_1, mh_1, cache = step_fn(
                    self.sharded_weights,
                    cache,
                    tok_arr,
                    jnp.asarray(cur_pos, dtype=jnp.int32),
                )
                next_tok = int(np.asarray(jnp.argmax(logits_1, axis=-1)).ravel()[0])
                step_ms = (time.perf_counter() - t_step0) * 1000.0
                step_times_ms.append(step_ms)
                generated.append(next_tok)
                cur_tok = next_tok
                cur_pos += 1
                if eos_token_id is not None:
                    if next_tok == int(eos_token_id):
                        break
                    if len(generated) >= 8 and len(generated) % 2 == 0:
                        tail_txt = self.tokenizer.decode(generated[-28:])
                        if _STOP_ANSWER_RE.search(tail_txt):
                            break
            else:
                # 1. Run DSpark 3-layer drafter to propose k tokens starting after `cur_tok`
                cand_block, _, _, cache = self.drafter.draft(
                    self.sharded_weights,
                    cache,
                    jnp.asarray([cur_tok], dtype=jnp.int32),
                    pending_mh,
                    pending_mh_start_pos,
                    confidence_threshold=confidence_threshold,
                )
                force_acc = None
                if force_accept_schedule is not None and step_idx < len(force_accept_schedule):
                    force_acc = int(force_accept_schedule[step_idx])

                # 2. Verify candidate block [1, 1 + k] on target model and roll back rejected slots
                res = verify_and_commit_block(
                    step_fn,
                    self.sharded_weights,
                    cache,
                    cand_block,
                    start_pos=cur_pos,
                    cfg=self.cfg,
                    force_num_accepted=force_acc,
                    stats=dspark_stats,
                )
                step_ms = (time.perf_counter() - t_step0) * 1000.0
                step_times_ms.append(step_ms)

                cache = res["cache"]
                pending_mh = res["accepted_main_hidden"]
                pending_mh_start_pos = cur_pos
                cur_pos = res["next_start_pos"]

                hit_eos = False
                for em_tok in res["emitted_tokens"]:
                    if len(generated) >= max_new_tokens:
                        break
                    generated.append(int(em_tok))
                    cur_tok = int(em_tok)
                    if eos_token_id is not None and int(em_tok) == int(eos_token_id):
                        hit_eos = True
                        break
                if not hit_eos and eos_token_id is not None and len(generated) >= 8:
                    tail_txt = self.tokenizer.decode(generated[-28:])
                    if _STOP_ANSWER_RE.search(tail_txt):
                        hit_eos = True
                step_idx += 1
                if hit_eos:
                    break

        decode_s = max(1e-9, time.perf_counter() - t_decode_start)
        decode_tokens = max(0, len(generated) - 1)
        tok_per_s = float(decode_tokens) / decode_s if decode_tokens > 0 else 0.0
        median_ms = float(np.median(step_times_ms)) if step_times_ms else 0.0
        p99_ms = float(np.percentile(step_times_ms, 99)) if step_times_ms else 0.0

        return {
            "prompt_ids": ids_np[0].tolist(),
            "generated_ids": generated,
            "full_ids": ids_np[0].tolist() + generated,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "tok_per_s": tok_per_s,
            "step_times_ms": step_times_ms,
            "median_step_ms": median_ms,
            "p99_step_ms": p99_ms,
            "dspark_stats": dspark_stats.to_dict(),
        }

    def benchmark_decode_step(
        self,
        batch_size: int = 1,
        *,
        start_pos: int = 64,
        num_warmup: int = 5,
        num_steps: int = 50,
        backend: str = "megakernel",
    ) -> dict[str, Any]:
        """Benchmark single-token decode latency at `batch_size` (`B in {1, 2, 4, 8}`)."""
        use_tpu_pallas = (
            backend.lower() == "megakernel"
            and jax.default_backend() != "cpu"
            and not bool(self.interpret)
            and self.cfg.n_layers >= 40
        )
        tpu_custom_calls: int | None = None

        if use_tpu_pallas:
            if self._tpu_packed_weights is None:
                self._tpu_packed_weights = prepare_tpu_megakernel_weights(
                    self.cfg, self.sharded_weights, self.mesh
                )
            bsz_key = int(batch_size)
            if bsz_key not in self._tpu_pallas_fns:
                self._tpu_pallas_fns[bsz_key] = make_tpu_pallas_megakernel_step(
                    self.cfg, self.mesh, batch_size=bsz_key
                )
            pallas_fn = self._tpu_pallas_fns[bsz_key]
            hlo_txt = pallas_fn.lower(*self._tpu_packed_weights).as_text()
            tpu_custom_calls = int(hlo_txt.count("tpu_custom_call"))

            t_comp0 = time.perf_counter()
            for _ in range(num_warmup):
                out = pallas_fn(*self._tpu_packed_weights)
                jax.block_until_ready(out)
            compile_and_warmup_s = time.perf_counter() - t_comp0

            step_times_ms: list[float] = []
            for _ in range(num_steps):
                t0 = time.perf_counter()
                out = pallas_fn(*self._tpu_packed_weights)
                jax.block_until_ready(out)
                step_times_ms.append((time.perf_counter() - t0) * 1000.0)
        else:
            cache = self.init_cache(batch_size=batch_size)
            step_fn = self.get_step_fn(backend=backend)
            tok = jnp.ones((batch_size, 1), dtype=jnp.int32)

            t_comp0 = time.perf_counter()
            for w_i in range(num_warmup):
                logits, _, cache = step_fn(
                    self.sharded_weights,
                    cache,
                    tok,
                    jnp.asarray(start_pos + w_i, dtype=jnp.int32),
                )
                jax.block_until_ready(logits)
            compile_and_warmup_s = time.perf_counter() - t_comp0

            step_times_ms = []
            base_pos = start_pos + num_warmup
            for s_i in range(num_steps):
                pos_val = base_pos + (s_i % 64)
                t0 = time.perf_counter()
                logits, _, cache = step_fn(
                    self.sharded_weights,
                    cache,
                    tok,
                    jnp.asarray(pos_val, dtype=jnp.int32),
                )
                jax.block_until_ready(logits)
                step_times_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.asarray(step_times_ms, dtype=np.float64)
        median_ms = float(np.median(arr))
        p10_ms = float(np.percentile(arr, 10))
        p90_ms = float(np.percentile(arr, 90))
        p99_ms = float(np.percentile(arr, 99))
        mean_ms = float(np.mean(arr))
        tok_per_s = float(batch_size) * 1000.0 / median_ms if median_ms > 0 else 0.0

        res_dict: dict[str, Any] = {
            "backend": backend,
            "batch_size": int(batch_size),
            "num_warmup": int(num_warmup),
            "num_steps": int(num_steps),
            "compile_and_warmup_s": compile_and_warmup_s,
            "median_step_ms": median_ms,
            "mean_step_ms": mean_ms,
            "p10_step_ms": p10_ms,
            "p90_step_ms": p90_ms,
            "p99_step_ms": p99_ms,
            "tok_per_s": tok_per_s,
            "step_times_ms": step_times_ms,
        }
        if tpu_custom_calls is not None:
            res_dict["tpu_custom_call_count"] = tpu_custom_calls
        return res_dict

    # -----------------------------------------------------------------------
    # Multi-host SPMD Broadcast Helpers (Coordinator Process 0 <-> Workers)
    # -----------------------------------------------------------------------

    def spmd_generate_greedy(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        *,
        use_dspark: bool = False,
        backend: str | None = None,
        eos_token_id: int | None = None,
    ) -> dict[str, Any]:
        """Broadcast generation request across all JAX processes and run `generate_greedy` in lockstep."""
        b_name = (backend or self.default_backend).lower()
        b_id = 0 if b_name == "megakernel" else 1
        eos_val = -1 if eos_token_id is None else int(eos_token_id)
        if jax.process_count() > 1:
            max_prompt = self.max_seq_len
            padded = np.zeros((max_prompt,), dtype=np.int32)
            plen = min(len(prompt_ids), max_prompt)
            padded[:plen] = np.asarray(prompt_ids[:plen], dtype=np.int32)
            header = np.asarray(
                [CMD_GENERATE, plen, int(max_new_tokens), int(use_dspark), b_id, eos_val],
                dtype=np.int32,
            )
            multihost_utils.broadcast_one_to_all(header)
            multihost_utils.broadcast_one_to_all(padded)
        return self.generate_greedy(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            use_dspark=use_dspark,
            backend=b_name,
            eos_token_id=eos_token_id,
        )

    def spmd_stop_workers(self) -> None:
        """Broadcast `CMD_STOP` so non-zero JAX processes exit `worker_serve_loop`."""
        if jax.process_count() > 1:
            header = np.asarray([CMD_STOP, 0, 0, 0, 0, 0], dtype=np.int32)
            multihost_utils.broadcast_one_to_all(header)

    def worker_serve_loop(self) -> None:
        """SPMD follower loop for `jax.process_index() != 0` while process 0 serves HTTP requests."""
        if jax.process_count() <= 1 or jax.process_index() == 0:
            return
        max_prompt = self.max_seq_len
        while True:
            dummy_hdr = np.zeros((6,), dtype=np.int32)
            hdr = np.asarray(multihost_utils.broadcast_one_to_all(dummy_hdr), dtype=np.int32)
            cmd = int(hdr[0])
            if cmd == CMD_STOP:
                break
            if cmd == CMD_GENERATE:
                plen, max_new, use_dsp, b_id, eos_val = (int(hdr[i]) for i in range(1, 6))
                dummy_buf = np.zeros((max_prompt,), dtype=np.int32)
                buf = np.asarray(multihost_utils.broadcast_one_to_all(dummy_buf), dtype=np.int32)
                prompt_ids = buf[:plen].tolist()
                b_name = "megakernel" if b_id == 0 else "xla"
                eos_id = None if eos_val < 0 else eos_val
                self.generate_greedy(
                    prompt_ids,
                    max_new_tokens=max_new,
                    use_dspark=bool(use_dsp),
                    backend=b_name,
                    eos_token_id=eos_id,
                )
