"""Validate and benchmark the Muse Spark 1.2 decode MEGAKERNEL (`musespark.decode_megakernel`)
on the REAL int4 weights against the XLA-reference oracle of `validate_musespark_prefill.py`.

One process, weights loaded once, tasks selected with `--tasks` (default: all):

    replay    Decode-only prompt replay (B=1, zero caches): every prompt id is fed as one kernel
              step from position 0, so the kernel writes and reads its own KV cache. After the
              last id the softcapped full-vocab logits are compared with `prompt{i}_logits.npy`
              (max |diff|, argmax, top-5 overlap) and, with `"aux_hidden"`, the residual stream
              after every layer (last prompt token) with `prompt0_hidden.npy[:, T-1]` (max |diff|
              and relative RMS per layer). The kernel-written K/V cache slots are also compared
              with the ones `musespark.prefill` writes for the same prompt.
    generate  Prefill (`musespark.prefill`, one cache row per prompt) + greedy kernel decode of
              48 tokens for all four prompts at B=4 and prompt 0 alone at B=1, through the demo's
              `make_chunk` (`--steps-per-call` steps per device call); the continuations are
              compared with `prompt{i}_gen.npy` (matching leading tokens) and printed verbatim.
    bench     Decode throughput for B in {1, 2, 4, 8} (every row holds prompt 0, ~150 tokens):
              ms/step, tok/s per row / aggregate after a warm-up call, `--bench-steps` timed
              steps; prefill time per bucket; the HBM-bandwidth floor and its utilisation.

Everything is logged to stdout and `logs/validate_musespark_decode_<UTC stamp>.log`, a JSON
summary goes next to it. Run only through the TPU launcher (exclusive 8-device lock):

    XLA_FLAGS=--xla_allow_excess_precision=false tpu_run.sh all \
        python scripts/validate_musespark_decode.py [--tasks replay,generate,bench]
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import validate_musespark_prefill as V  # noqa: E402  (loader / prompt rendering / oracle paths)
from demo_musespark import make_chunk  # noqa: E402
from musespark import decode_megakernel as dk  # noqa: E402
from musespark import layout, sampling  # noqa: E402
from musespark import load as ms_load  # noqa: E402
from musespark.prefill import make_prefill, pad_prompt  # noqa: E402

TP = V.TP
DEFAULT_LOG_DIR = ROOT / "logs"
BENCH_BATCHES = (1, 2, 4, 8)
PREFILL_BUCKETS = (64, 192, 512, 1024, 2048)
HBM_BYTES_PER_S = 3.2e12  # TPU7x per-core HBM bandwidth (tpu7x_hw_report.md)
MIB = float(1 << 20)


# --------------------------------------------------------------------------------------
# HBM floor
# --------------------------------------------------------------------------------------
def hbm_floor_ms(cfg, batch, tp=TP):
    """Bytes one core must read per decode step and the implied floor in ms.

    Dense per layer (bf16, per rank): q, kv, gate, o, pre, post, router hi/lo; experts: the
    routed experts' int4 gate_up + down with their scales, at most `top_k * batch` distinct
    (the upper bound, so the floor at B > 1 is slightly pessimistic); lm_head per rank.
    """
    shapes = layout.rank_shapes(cfg, tp)

    def nbytes(name):
        shape, dtype = shapes[name]
        item = 0.5 if jnp.dtype(dtype) == jnp.dtype(jnp.int4) else jnp.dtype(dtype).itemsize
        return int(np.prod(shape)) * item

    dense = sum(nbytes(n) for n in layout.STREAMED_FAMILIES if n != "lm_head")  # all layers
    per_expert = sum(nbytes(n) for n in layout.EXPERT_FAMILIES) / (cfg.layers * cfg.experts)
    experts = per_expert * min(cfg.top_k * batch, cfg.experts) * cfg.layers
    lm_head = nbytes("lm_head")
    total = dense + experts + lm_head
    return {
        "dense_mib": dense / MIB,
        "experts_mib": experts / MIB,
        "lm_head_mib": lm_head / MIB,
        "total_gb": total / 1e9,
        "floor_ms": 1e3 * total / HBM_BYTES_PER_S,
    }


# --------------------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------------------
class Harness:
    """Weights, tokenizer, oracle and compiled programs (cached by configuration)."""

    def __init__(self, args, log):
        self.args, self.log = args, log
        self.context = args.context
        self.ref = Path(args.ref)
        devices = jax.devices()
        if len(devices) < TP:
            raise RuntimeError(f"need {TP} TPU devices, found {len(devices)}")
        self.mesh = jax.sharding.Mesh(np.asarray(devices[:TP]), ("tp",))
        self.doc = ms_load.read_layout(args.weights)
        self.cfg = ms_load.config_from_layout(self.doc)
        self.tokenizer = ms_load.load_tokenizer(args.checkpoint)
        self.prompts = self.load_prompts()
        log(f"loading weights from {args.weights} ({self.doc['total_bytes'] / 1e9:.0f} GB)")
        t0 = time.perf_counter()
        self.weights = ms_load.load_presharded(self.mesh, args.weights, self.cfg, log=lambda _: None)
        self.load_seconds = time.perf_counter() - t0
        log(f"weights resident in {self.load_seconds:.0f} s")
        self.decoders = {}
        self.chunks = {}
        self.prefills = {}
        self.prefill_compiled = set()
        self.argmax = jax.jit(lambda lg: jnp.argmax(sampling.mask_unused(lg, self.cfg)))
        self.replicated = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

    def replicate(self, x, dtype=jnp.int32):
        """Place a small array replicated on the mesh: the sharding the decode programs return,
        so the first call already uses the executable of every later one (jit specialises on
        input shardings; a single-device `jnp.asarray` input would compile a second variant)."""
        return jax.device_put(jnp.asarray(x, dtype), self.replicated)

    # --- oracle -------------------------------------------------------------------------
    def load_prompts(self):
        """`[(ids, ref_logits, ref_gen)]` for every oracle prompt; the ids are re-rendered and
        checked against the saved ones."""
        specs = V.prompt_specs()
        out = []
        for i, (name, kind, text) in enumerate(specs):
            path = self.ref / f"prompt{i}_ids.npy"
            if not path.is_file():
                break
            ids, _ = V.encode_prompt(self.cfg, self.tokenizer, kind, text)
            saved = np.load(path).tolist()
            if ids != saved:
                raise RuntimeError(f"prompt {i}: rendered ids differ from the oracle {path}")
            logits = np.load(self.ref / f"prompt{i}_logits.npy").astype(np.float32)
            gen = np.load(self.ref / f"prompt{i}_gen.npy").astype(np.int32)
            out.append((ids, logits, gen))
            self.log(f"oracle prompt {i} ({name}): {len(ids)} ids, {len(gen)} reference tokens")
        if not out:
            raise RuntimeError(f"no oracle under {self.ref} (run validate_musespark_prefill.py)")
        return out

    # --- programs -------------------------------------------------------------------------
    def decoder(self, batch, *, return_logits=False, aux=False):
        key = (batch, return_logits, aux)
        if key not in self.decoders:
            options = frozenset({"aux_hidden"} if aux else set())
            self.log(f"building decode program B={batch} logits={return_logits} aux={aux}")
            self.decoders[key] = dk.make_decode(
                self.mesh, self.cfg, self.context, batch, greedy=True,
                return_logits=return_logits, options=options,
            )
        return self.decoders[key]

    def chunk(self, batch):
        """Greedy `make_chunk` program (`--steps-per-call` kernel steps per device call)."""
        if batch not in self.chunks:
            self.chunks[batch] = make_chunk(
                self.decoder(batch), self.args.steps_per_call, self.cfg, greedy=True, top_k=None
            )
        return self.chunks[batch]

    def run_chunk(self, batch, caches, cur, pos):
        """`(next cur [B], caches, out [steps, B])`; seconds are measured by the caller."""
        cur, caches, out, _ = self.chunk(batch)(
            self.weights, caches, self.replicate(cur), self.replicate(pos),
            jax.random.key(0), jnp.float32(0.0), jnp.float32(1.0),
        )
        return cur, caches, out

    def prefill(self, caches, ids, row):
        """Prefill `ids` into cache row `row`: (softcapped logits [V], caches, seconds, bucket,
        compiled-now)."""
        batch = caches["k_cache"].shape[2]
        if batch not in self.prefills:
            self.prefills[batch] = make_prefill(self.mesh, self.cfg, self.context, TP)
        tokens, length = pad_prompt(self.cfg, ids)
        bucket = int(tokens.shape[0])
        first = (batch, bucket) not in self.prefill_compiled
        t0 = time.perf_counter()
        logits, caches = self.prefills[batch](self.weights, caches, tokens, length, row)
        logits.block_until_ready()
        seconds = time.perf_counter() - t0
        if first:
            self.prefill_compiled.add((batch, bucket))
            self.log(f"prefill B={batch} bucket {bucket}: compile + first run {seconds:.1f} s")
        return logits, caches, seconds, bucket, first

    def zero_caches(self, batch):
        return ms_load.zero_caches(self.mesh, self.cfg, batch, self.context)

    def text(self, ids):
        return self.tokenizer.decode([int(t) for t in ids])


# --------------------------------------------------------------------------------------
# task 1: decode-only prompt replay
# --------------------------------------------------------------------------------------
def compare_logits(cfg, got, ref):
    """Dict of max |diff|, argmax equality and top-5 overlap on the used vocabulary."""
    n = cfg.vocab_used
    got, ref = np.asarray(got, np.float32)[:n], np.asarray(ref, np.float32)[:n]
    diff = np.abs(got - ref)
    top_got = np.argsort(-got, kind="stable")[:5]
    top_ref = np.argsort(-ref, kind="stable")[:5]
    margin = float(np.sort(ref)[-1] - np.sort(ref)[-2])
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
        "argmax_got": int(top_got[0]),
        "argmax_ref": int(top_ref[0]),
        "argmax_equal": bool(top_got[0] == top_ref[0]),
        "top5_got": top_got.tolist(),
        "top5_ref": top_ref.tolist(),
        "top5_overlap": int(len(set(top_got.tolist()) & set(top_ref.tolist()))),
        "ref_top1_margin": margin,
        "finite": bool(np.all(np.isfinite(got))),
    }


def replay_steps(h: Harness, ids, batch=1, save_aux=None):
    """Feed `ids` one per kernel step from position 0 into `batch` identical rows (zero caches).

    Returns (logits [B, V] of the last step, aux [L+1, B, H] of the last step, caches,
    teacher-forced top-1 hits of row 0, per-step seconds). With `save_aux` the residual streams
    of every step are saved as `[L+1, T, H]` (row 0; the oracle's `prompt0_hidden.npy` layout).
    """
    T = len(ids)
    decode = h.decoder(batch, return_logits=True, aux=True)
    caches = h.zero_caches(batch)
    hits, times = 0, []
    streams = np.zeros((h.cfg.layers + 1, T, h.cfg.hidden), np.float32) if save_aux else None
    for i, tok in enumerate(ids):
        ts = time.perf_counter()
        nxt, logits, caches, aux = decode(
            h.weights, caches, h.replicate(np.full((batch,), tok)), h.replicate(np.full((batch,), i))
        )
        nxt.block_until_ready()
        times.append(time.perf_counter() - ts)
        if i == 0:
            h.log(f"replay: first kernel call (compile + run) {times[0]:.1f} s")
        if i + 1 < T:
            hits += int(nxt[0]) == ids[i + 1]
        if streams is not None:
            streams[:, i] = np.asarray(aux[:, 0], np.float32)
    if streams is not None:
        np.save(save_aux, streams)
        h.log(f"replay: kernel residual streams of all {T} tokens saved to {save_aux}")
    return np.asarray(logits, np.float32), np.asarray(aux, np.float32), caches, hits, times


def task_replay(h: Harness, prompt_index, batches=(1,), save_aux_dir=None):
    cfg, log = h.cfg, h.log
    ids, ref_logits, _ = h.prompts[prompt_index]
    T = len(ids)
    hidden_path = h.ref / f"prompt{prompt_index}_hidden.npy"
    ref_hidden = None
    if hidden_path.is_file():
        ref_hidden = np.load(hidden_path, mmap_mode="r")[:, T - 1].astype(np.float32)  # [L+1, H]
    log(f"=== replay prompt {prompt_index}: {T} kernel steps from position 0 (B=1, aux taps)")
    save_aux = None
    if save_aux_dir:
        Path(save_aux_dir).mkdir(parents=True, exist_ok=True)
        save_aux = Path(save_aux_dir) / f"kernel_prompt{prompt_index}_hidden.npy"
    t0 = time.perf_counter()
    logits_b, aux_b, caches, top1_hits, times = replay_steps(h, ids, 1, save_aux)
    total = time.perf_counter() - t0
    logits, aux = logits_b[0], aux_b[:, 0]
    steady = float(np.median(times[1:])) if len(times) > 1 else float("nan")
    log(
        f"replay: {T} steps in {total:.1f} s (median step incl. dispatch/aux {steady * 1e3:.1f} "
        f"ms); teacher-forced top-1 hits on the prompt {top1_hits}/{T - 1}"
    )
    result = {"prompt": prompt_index, "tokens": T, "teacher_forced_top1": top1_hits}
    # --- logits ------------------------------------------------------------------------------
    cmp = compare_logits(cfg, logits, ref_logits)
    result["logits"] = cmp
    tk = h.tokenizer
    log(
        f"replay: logits max |diff| {cmp['max_abs_diff']:.4f} (mean {cmp['mean_abs_diff']:.5f}, "
        f"rms {cmp['rms_diff']:.5f}), argmax {cmp['argmax_got']} {tk.decode([cmp['argmax_got']])!r}"
        f" vs ref {cmp['argmax_ref']} {tk.decode([cmp['argmax_ref']])!r}: "
        f"{'EQUAL' if cmp['argmax_equal'] else 'DIFFERENT'}; top-5 overlap {cmp['top5_overlap']}/5"
        f" (kernel {cmp['top5_got']}, ref {cmp['top5_ref']}; ref top-1 margin "
        f"{cmp['ref_top1_margin']:.3f})"
    )
    # --- per-layer residual stream ---------------------------------------------------------------
    if ref_hidden is not None:
        rows = []
        for l in range(cfg.layers + 1):
            d = aux[l] - ref_hidden[l]
            rms_ref = float(np.sqrt(np.mean(ref_hidden[l] ** 2)))
            rows.append(
                {
                    "layer": l,
                    "max_abs_diff": float(np.abs(d).max()),
                    "rel_rms": float(np.sqrt(np.mean(d**2)) / max(rms_ref, 1e-30)),
                    "ref_rms": rms_ref,
                    "exact": float(np.mean(d == 0)),
                }
            )
        result["hidden"] = rows
        log("replay: residual stream of the last prompt token vs prompt0_hidden.npy per layer")
        log("  layer  max|diff|   rel RMS   ref RMS   exact")
        for r in rows:
            log(
                f"  {r['layer']:5d}  {r['max_abs_diff']:9.4f}  {r['rel_rms']:8.2e}  "
                f"{r['ref_rms']:7.3f}  {r['exact']:.3f}"
            )
        worst = max(rows[1:], key=lambda r: r["rel_rms"])
        log(
            f"replay: tap 0 exact {rows[0]['exact'] == 1.0}; worst relative RMS "
            f"{worst['rel_rms']:.2e} at layer {worst['layer']}; final layer rel RMS "
            f"{rows[-1]['rel_rms']:.2e}"
        )
    # --- KV cache vs the XLA prefill ---------------------------------------------------------
    pf_logits, pf_caches, seconds, bucket, _ = h.prefill(h.zero_caches(1), ids, 0)
    pf_cmp = compare_logits(cfg, np.asarray(pf_logits), ref_logits)
    log(
        f"replay: XLA prefill (bucket {bucket}, {seconds:.2f} s) logits vs oracle max |diff| "
        f"{pf_cmp['max_abs_diff']:.4f}; kernel vs prefill max |diff| "
        f"{np.abs(logits[: cfg.vocab_used] - np.asarray(pf_logits)[: cfg.vocab_used]).max():.4f}"
    )
    result["prefill_logits"] = pf_cmp
    cache_rows = {}
    for name in ("k_cache", "v_cache"):
        mine = np.asarray(caches[name][:, :, 0, :T].astype(jnp.float32))  # [tp, L, T, lanes]
        theirs = np.asarray(pf_caches[name][:, :, 0, :T].astype(jnp.float32))
        d = np.abs(mine - theirs)
        per_layer = d.max(axis=(0, 2, 3))
        exact = (d == 0).mean(axis=(0, 2, 3))
        scale = np.abs(theirs).max(axis=(0, 2, 3))
        cache_rows[name] = {
            "max_abs_diff": float(d.max()),
            "exact_fraction": float((d == 0).mean()),
            "per_layer_max": per_layer.tolist(),
            "per_layer_exact": exact.tolist(),
        }
        log(
            f"replay: {name} kernel vs prefill over {T} slots: max |diff| {d.max():.4f} "
            f"(values up to {scale.max():.2f}), exact fraction {(d == 0).mean():.4f}; "
            f"per-layer max {np.round(per_layer, 3).tolist()}"
        )
        untouched = np.asarray(caches[name][:, :, 0, T : T + 64])
        log(f"replay: {name} slots {T}..{T + 63} untouched: {not np.any(untouched)}")
    result["cache_vs_prefill"] = cache_rows
    del caches, pf_caches
    # --- batch consistency: the same replay in every row of a wider batch --------------------
    result["batch_consistency"] = []
    for batch in batches:
        if batch == 1:
            continue
        log(f"replay: prompt {prompt_index} again at B={batch} (identical rows)")
        logits_w, aux_w, caches_w, hits_w, _ = replay_steps(h, ids, batch)
        n = cfg.vocab_used
        vs_b1 = float(np.abs(logits_w[:, :n] - logits[None, :n]).max(axis=1).max())
        rows = float(np.abs(logits_w[:, :n] - logits_w[:1, :n]).max())
        hid_rows = float(np.abs(aux_w - aux_w[:, :1]).max())
        hid_b1 = float(np.abs(aux_w[:, 0] - aux).max())
        argmax_w = np.argmax(logits_w[:, :n], axis=1).tolist()
        log(
            f"replay: B={batch} vs B=1: logits max |diff| {vs_b1:.4f}, argmax {argmax_w} "
            f"(B=1: {cmp['argmax_got']}); rows vs row 0: logits max |diff| {rows:.4f}, "
            f"residual max |diff| {hid_rows:.4f}; residual row 0 vs B=1 max |diff| {hid_b1:.4f}; "
            f"teacher-forced top-1 hits {hits_w}/{T - 1}"
        )
        result["batch_consistency"].append(
            {"batch": batch, "logits_vs_b1": vs_b1, "logits_rows_vs_row0": rows,
             "hidden_rows_vs_row0": hid_rows, "hidden_row0_vs_b1": hid_b1, "argmax": argmax_w}
        )
        del caches_w
    return result


# --------------------------------------------------------------------------------------
# task 2: prefill + kernel generation
# --------------------------------------------------------------------------------------
def task_generate(h: Harness, prompt_indices, max_tokens=48):
    cfg, log, args = h.cfg, h.log, h.args
    batch = len(prompt_indices)
    log(f"=== generate: prompts {list(prompt_indices)} at B={batch}, {max_tokens} tokens")
    caches = h.zero_caches(batch)
    first = np.full((batch,), cfg.pad, np.int32)
    pos = np.zeros((batch,), np.int32)
    prefill_seconds = 0.0
    for b, pi in enumerate(prompt_indices):
        ids, ref_logits, _ = h.prompts[pi]
        logits, caches, seconds, bucket, compiled = h.prefill(caches, ids, b)
        prefill_seconds += seconds
        first[b] = int(h.argmax(logits))
        pos[b] = len(ids)
        d = np.abs(np.asarray(logits)[: cfg.vocab_used] - ref_logits[: cfg.vocab_used]).max()
        log(f"generate: row {b} prompt {pi}: prefill bucket {bucket} {seconds:.2f} s, "
            f"logits vs oracle max |diff| {d:.4f}, first token {first[b]}")
    generated = [[int(t)] for t in first]
    cur, pos_dev = h.replicate(first), h.replicate(pos)
    steps = 0
    t0 = time.perf_counter()
    while len(generated[0]) < max_tokens:
        tc = time.perf_counter()
        cur, caches, out = h.run_chunk(batch, caches, cur, pos_dev)
        out = np.asarray(out)
        log(f"generate: chunk of {args.steps_per_call} steps {time.perf_counter() - tc:.2f} s"
            f"{' (compile + run)' if steps == 0 else ''}")
        for b in range(batch):
            generated[b].extend(int(t) for t in out[:, b])
        pos_dev = pos_dev + args.steps_per_call
        steps += args.steps_per_call
    decode_seconds = time.perf_counter() - t0
    log(f"generate: {steps} kernel steps in {decode_seconds:.2f} s")
    results = []
    for b, pi in enumerate(prompt_indices):
        ids, _, ref_gen = h.prompts[pi]
        got = generated[b][: len(ref_gen)]
        compared = min(len(got), len(ref_gen))
        match = 0
        while match < compared and got[match] == int(ref_gen[match]):
            match += 1
        text = h.text(got)
        note = ""
        if match < compared:
            note = (
                f" (first mismatch at {match}: kernel {got[match]} {h.text([got[match]])!r} vs "
                f"ref {int(ref_gen[match])} {h.text([int(ref_gen[match])])!r})"
            )
        log(f"generate: row {b} prompt {pi}: {match}/{compared} leading tokens match the oracle{note}")
        log(f"generate: row {b} prompt {pi} kernel text: {text!r}")
        if match < compared:
            log(f"generate: row {b} prompt {pi} oracle text: {h.text(ref_gen)!r}")
        results.append(
            {
                "prompt": pi,
                "row": b,
                "batch": batch,
                "matching_leading_tokens": match,
                "compared_tokens": compared,
                "reference_tokens": int(len(ref_gen)),
                "generated_ids": got,
                "generated_text": text,
                "reference_text": h.text(ref_gen),
            }
        )
    del caches
    return results


# --------------------------------------------------------------------------------------
# task 3: benchmark
# --------------------------------------------------------------------------------------
def task_bench(h: Harness):
    cfg, log, args = h.cfg, h.log, h.args
    ids = h.prompts[0][0]
    results = []
    for batch in BENCH_BATCHES:
        log(f"=== bench B={batch}: context {h.context}, prompt of {len(ids)} tokens in every row")
        caches = h.zero_caches(batch)
        prefill_ms = {}
        if batch == 1:  # bucket timings: B-independent enough, measured once
            for bucket in PREFILL_BUCKETS:
                if bucket + args.steps_per_call > h.context:
                    continue
                filler = (list(ids) * (bucket // len(ids) + 1))[:bucket]
                _, caches, _, _, _ = h.prefill(caches, filler, 0)  # compile + first run
                _, caches, seconds, _, _ = h.prefill(caches, filler, 0)
                prefill_ms[bucket] = seconds * 1e3
            log("bench: prefill ms per bucket (B=1): " + ", ".join(f"T={k}: {v:.0f}" for k, v in prefill_ms.items()))
        prompt_bucket_ms = []
        for b in range(batch):
            _, caches, seconds, bucket, compiled = h.prefill(caches, ids, b)
            if not compiled:
                prompt_bucket_ms.append(seconds * 1e3)
        if not prompt_bucket_ms:
            _, caches, seconds, bucket, _ = h.prefill(caches, ids, 0)
            prompt_bucket_ms.append(seconds * 1e3)
        pos = h.replicate(np.full((batch,), len(ids)))
        cur = h.replicate(np.full((batch,), int(ids[-1])))
        warm = []
        for i in range(2):  # the first call compiles, the second still carries some overhead
            t0 = time.perf_counter()
            cur, caches, out = h.run_chunk(batch, caches, cur, pos + i * args.steps_per_call)
            out.block_until_ready()
            warm.append(time.perf_counter() - t0)
        log(f"bench: warm-up chunks {warm[0]:.2f} s (compile + run), {warm[1]:.3f} s")
        calls = max(1, -(-args.bench_steps // args.steps_per_call))
        per_call = []
        for i in range(calls):
            t0 = time.perf_counter()
            cur, caches, out = h.run_chunk(batch, caches, cur, pos + (i + 2) * args.steps_per_call)
            out.block_until_ready()
            per_call.append(time.perf_counter() - t0)
        steps = calls * args.steps_per_call
        seconds = sum(per_call)
        ms = 1e3 * seconds / steps
        best_ms = 1e3 * min(per_call) / args.steps_per_call
        worst_ms = 1e3 * max(per_call) / args.steps_per_call
        floor = hbm_floor_ms(cfg, batch)
        row = {
            "batch": batch,
            "steps": steps,
            "steps_per_call": args.steps_per_call,
            "ms_per_step": ms,
            "best_call_ms_per_step": best_ms,
            "worst_call_ms_per_step": worst_ms,
            "per_call_seconds": per_call,
            "tok_s_per_row": 1e3 / ms,
            "tok_s_aggregate": batch * 1e3 / ms,
            "prefill_prompt_bucket": bucket,
            "prefill_prompt_ms": float(np.median(prompt_bucket_ms)),
            "prefill_bucket_ms": prefill_ms,
            "hbm_floor": floor,
            "hbm_utilisation": floor["floor_ms"] / ms,
        }
        results.append(row)
        log(
            f"bench: B={batch}: {steps} steps in {seconds:.3f} s = {ms:.3f} ms/step "
            f"(best call {best_ms:.3f}, worst {worst_ms:.3f}), {1e3 / ms:.1f} tok/s per row, "
            f"{batch * 1e3 / ms:.1f} tok/s aggregate; prefill T={bucket} {np.median(prompt_bucket_ms):.0f} ms; "
            f"HBM floor {floor['floor_ms']:.2f} ms ({floor['total_gb']:.2f} GB: dense "
            f"{floor['dense_mib']:.0f} MiB + experts {floor['experts_mib']:.0f} MiB + lm_head "
            f"{floor['lm_head_mib']:.0f} MiB) -> {100 * floor['floor_ms'] / ms:.0f}% of the floor"
        )
        del caches
    log(f"bench summary (context {h.context}, {args.steps_per_call}-step calls):")
    log(f"  B   ms/step  tok/s/row  tok/s aggr  floor ms  util%  "
        f"prefill(T={results[0]['prefill_prompt_bucket']}) ms")
    for r in results:
        log(
            f"  {r['batch']:d}  {r['ms_per_step']:8.3f}  {r['tok_s_per_row']:9.1f}  "
            f"{r['tok_s_aggregate']:10.1f}  {r['hbm_floor']['floor_ms']:8.2f}  "
            f"{100 * r['hbm_utilisation']:5.0f}  {r['prefill_prompt_ms']:6.0f}"
        )
    return results


# --------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", default=V.DEFAULT_WEIGHTS)
    parser.add_argument("--checkpoint", default=V.DEFAULT_CHECKPOINT)
    parser.add_argument("--ref", default=V.DEFAULT_OUT, help="oracle directory")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--tasks", default="replay,generate,bench")
    parser.add_argument("--replay-prompts", default="0,3")
    parser.add_argument(
        "--replay-batches", default="1,4",
        help="also replay every row of these batch sizes with the same prompt (consistency)",
    )
    parser.add_argument(
        "--save-aux", default=None, metavar="DIR",
        help="save the kernel residual streams of every replayed token under DIR",
    )
    parser.add_argument("--steps-per-call", type=int, default=16)
    parser.add_argument("--bench-steps", type=int, default=128)
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    args = parser.parse_args(argv)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"validate_musespark_decode_{stamp}.log"
    started = time.perf_counter()
    handle = log_path.open("w")

    def log(message):
        line = f"[{time.perf_counter() - started:6.1f}s] {message}"
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    log(f"log: {log_path}; args: {vars(args)}")
    h = Harness(args, log)
    summary = {"args": vars(args), "load_seconds": h.load_seconds, "jax": jax.__version__}
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if "replay" in tasks:
        batches = tuple(int(b) for b in args.replay_batches.split(",") if b.strip())
        summary["replay"] = [
            task_replay(h, int(p), batches, args.save_aux)
            for p in args.replay_prompts.split(",")
            if p.strip()
        ]
    if "generate" in tasks:
        summary["generate"] = task_generate(h, tuple(range(len(h.prompts)))) + task_generate(
            h, (0,)
        )
    if "bench" in tasks:
        summary["bench"] = task_bench(h)
    json_path = log_dir / f"validate_musespark_decode_{stamp}.json"
    json_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    log(f"done in {time.perf_counter() - started:.0f} s; summary {json_path}")
    handle.close()
    return summary


if __name__ == "__main__":
    main()
