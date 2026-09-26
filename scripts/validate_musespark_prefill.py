"""Validate the Muse Spark 1.2 XLA prefill (`musespark.prefill`) on the REAL int4 weights.

Loads the pre-sharded container, builds the prefill for one context, and for a few chat
prompts (rendered with `musespark.chat.render_chat`) plus one raw-text prompt runs GREEDY
generation by "prefill-as-decode": the prompt plus the tokens generated so far is prefilled
again for every new token (one executable per 64-token bucket). For every prompt it prints
tokens/s, the top-5 next-token candidates after the prompt and the decoded continuation, and
saves an oracle for the kernel tests under `--out`:

    prompt{i}_ids.npy      int32 [T]      prompt ids
    prompt{i}_logits.npy   float32 [V]    softcapped full-vocab logits after the prompt (unmasked)
    prompt{i}_gen.npy      int32 [n]      greedy continuation (stop token included when hit)
    prompt0_hidden.npy     float32 [L+1, Tp, H]  residual stream taps of the FIRST prompt
    summary.json           prompts, texts, top-5, timings, config, load time

Run only through the TPU launcher (exclusive 8-device lock), e.g.

    XLA_FLAGS=--xla_allow_excess_precision=false tpu_run.sh all \
        python scripts/validate_musespark_prefill.py

`--deadline` stops generation early so one process never holds the devices for too long.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from musespark import chat, sampling  # noqa: E402
from musespark import load as ms_load  # noqa: E402
from musespark.config import Config  # noqa: E402
from musespark.prefill import make_prefill, pad_prompt  # noqa: E402

DEFAULT_WEIGHTS = "/filestore/weights/muse-spark-tp8-int4"
DEFAULT_CHECKPOINT = "/filestore/weights/Muse-Spark-1.2-816B-A42B-open"
DEFAULT_OUT = (
    "/filestore/tmp/claude-0/-filestore-weights/bc41d5f3-5fda-4ab7-9ca1-48e0c99553cc/"
    "scratchpad/real_ref"
)
TP = 8
DATE = "2026-09-26"  # fixed so the rendered prompts (and the oracle) are reproducible
REASONING_EFFORT = "medium"
CHAT_PROMPTS = (
    "What is 2+2?",
    "Write a haiku about the ocean.",
    "Name the capital of France and explain briefly.",
)
RAW_PROMPTS = ("The quick brown fox",)
PARAGRAPH = (
    "The Pacific Ocean is the largest and deepest of Earth's five oceans. It extends from "
    "the Arctic Ocean in the north to the Southern Ocean in the south and is bounded by Asia "
    "and Australia in the west and the Americas in the east."
)


def prompt_specs():
    """`[(name, kind, text)]` in the order they are run (prompt index = position)."""
    return [("chat", "chat", t) for t in CHAT_PROMPTS] + [("raw", "raw", t) for t in RAW_PROMPTS]


def encode_prompt(cfg, tokenizer, kind, text):
    if kind == "raw":
        return [cfg.bos] + list(tokenizer.encode(text)), text
    rendered = chat.render_chat(text, date=DATE, reasoning_effort=REASONING_EFFORT)
    return list(tokenizer.encode(rendered)), rendered


def top_candidates(cfg, tokenizer, logits, n=5):
    """`[(id, token text, prob)]` of the `n` most likely next tokens (unused ids masked)."""
    masked = np.asarray(sampling.mask_unused(jnp.asarray(logits), cfg))
    z = masked - masked.max()
    p = np.exp(z) / np.exp(z).sum()
    ids = np.argsort(-masked, kind="stable")[:n]
    return [(int(i), tokenizer.decode([int(i)]), float(p[i])) for i in ids]


class Runner:
    """Prefill-as-decode greedy generation over one KV-cache row."""

    def __init__(self, mesh, cfg, weights, context, log):
        self.mesh, self.cfg, self.weights, self.context, self.log = mesh, cfg, weights, context, log
        self.caches = ms_load.zero_caches(mesh, cfg, 1, context)
        self.programs = {False: make_prefill(mesh, cfg, context, TP, taps=False), True: None}
        self.compiled = set()  # (bucket, taps) already compiled
        self.argmax = jax.jit(lambda lg: jnp.argmax(sampling.mask_unused(lg, cfg)))

    def step(self, ids, taps=False):
        """One prefill of `ids`: (logits [V] np.float32, seconds, hidden or None)."""
        if taps and self.programs[True] is None:
            self.programs[True] = make_prefill(self.mesh, self.cfg, self.context, TP, taps=True)
        tokens, length = pad_prompt(self.cfg, ids)
        key = (tokens.shape[0], taps)
        started = time.perf_counter()
        out = self.programs[taps](self.weights, self.caches, tokens, length, 0)
        jax.block_until_ready(out)
        seconds = time.perf_counter() - started
        logits, self.caches = out[0], out[1]
        hidden = out[2] if taps else None
        if key not in self.compiled:
            self.compiled.add(key)
            self.log(f"prefill bucket {key[0]} taps={taps}: compile + first run {seconds:.1f} s")
        return logits, seconds, hidden

    def generate(self, ids, max_tokens, deadline):
        """Greedy continuation of `ids` (stops on STOP_IDS / max_tokens / deadline).

        Returns (generated ids, first logits [V], list of per-call seconds, stop reason)."""
        ids = list(ids)
        generated, times, first_logits, reason = [], [], None, "length"
        for _ in range(max_tokens):
            logits, seconds, _ = self.step(ids)
            times.append(seconds)
            if first_logits is None:
                first_logits = np.asarray(logits, np.float32)
            nxt = int(self.argmax(logits))
            generated.append(nxt)
            ids.append(nxt)
            if nxt in chat.STOP_IDS:
                reason = "stop"
                break
            if time.perf_counter() > deadline:
                reason = "deadline"
                break
        return generated, first_logits, times, reason


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--prompts", type=int, default=None, help="only the first N prompts")
    parser.add_argument("--no-taps", action="store_true", help="skip the residual-stream taps")
    parser.add_argument(
        "--deadline", type=float, default=17 * 60, help="stop generating after this many seconds"
    )
    parser.add_argument(
        "--teacher-force",
        action="store_true",
        help="also score PARAGRAPH token by token (one prefill per prefix): mean log-prob",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    deadline = started + args.deadline

    def log(message):
        print(f"[{time.perf_counter() - started:6.1f}s] {message}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    if len(devices) < TP:
        raise RuntimeError(f"need {TP} TPU devices, found {len(devices)}")
    mesh = jax.sharding.Mesh(np.asarray(devices[:TP]), ("tp",))
    doc = ms_load.read_layout(args.weights)
    cfg = ms_load.config_from_layout(doc)
    # check (b): the container config equals the checkpoint config
    ckpt_cfg = Config.from_checkpoint(args.checkpoint)
    log(f"config from layout == Config.from_checkpoint: {cfg == ckpt_cfg}")
    if cfg != ckpt_cfg:
        for k in cfg.__dict__:
            if getattr(cfg, k) != getattr(ckpt_cfg, k):
                log(f"  config mismatch {k}: container {getattr(cfg, k)!r} ckpt {getattr(ckpt_cfg, k)!r}")
    tokenizer = ms_load.load_tokenizer(args.checkpoint)

    # check (a): tokenizer round trip on the rendered prompts
    specs = prompt_specs()[: args.prompts]
    prompts = []
    for name, kind, text in specs:
        ids, rendered = encode_prompt(cfg, tokenizer, kind, text)
        round_trip = tokenizer.decode(ids)
        expect = rendered if kind == "chat" else chat.BOS + text
        log(
            f"prompt {len(prompts)} ({name}): {len(ids)} ids, first {ids[:4]}, last {ids[-3:]}, "
            f"round trip ok: {round_trip == expect}"
        )
        prompts.append((name, kind, text, ids, rendered))
    log(f"rendered prompt 0:\n{prompts[0][4]}")

    log(f"loading weights from {args.weights} ({doc['total_bytes'] / 1e9:.0f} GB)")
    t0 = time.perf_counter()
    weights = ms_load.load_presharded(mesh, args.weights, cfg, log=log)
    load_seconds = time.perf_counter() - t0
    log(f"weights resident in {load_seconds:.0f} s")

    runner = Runner(mesh, cfg, weights, args.context, log)
    summary = {
        "weights": args.weights,
        "checkpoint": args.checkpoint,
        "revision": doc.get("revision"),
        "context": args.context,
        "date": DATE,
        "reasoning_effort": REASONING_EFFORT,
        "config_matches_checkpoint": cfg == ckpt_cfg,
        "load_seconds": load_seconds,
        "prompts": [],
    }

    def save_summary():
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    for i, (name, kind, text, ids, rendered) in enumerate(prompts):
        log(f"=== prompt {i} ({name}): {text!r} ({len(ids)} ids)")
        if i == 0 and not args.no_taps:
            logits_t, seconds, hidden = runner.step(ids, taps=True)
            hidden = np.asarray(hidden, np.float32)
            np.save(out_dir / f"prompt{i}_hidden.npy", hidden)
            log(
                f"taps: hidden {hidden.shape} saved, per-layer RMS of the first {len(ids)} rows: "
                f"{np.round(np.sqrt((hidden[:, :len(ids)] ** 2).mean(axis=(1, 2))), 3).tolist()}"
            )
            summary["taps_shape"] = list(hidden.shape)
            logits_t = np.asarray(logits_t, np.float32)
        generated, first_logits, times, reason = runner.generate(ids, args.max_tokens, deadline)
        if i == 0 and not args.no_taps:
            d = float(np.abs(first_logits - logits_t).max())
            log(f"taps program vs plain program logits: max |diff| {d:.3e}")
            summary["taps_logit_max_diff"] = d
        top5 = top_candidates(cfg, tokenizer, first_logits)
        text_out = tokenizer.decode(generated)
        steady = times[1:] if len(times) > 1 else times  # first call of a bucket compiles
        tps = len(generated) / max(sum(times), 1e-9)
        steady_tps = len(steady) / max(sum(steady), 1e-9) if steady else float("nan")
        log(f"top-5 after the prompt: {[(t, i_, round(p, 4)) for i_, t, p in top5]}")
        log(
            f"generated {len(generated)} tokens ({reason}) in {sum(times):.1f} s: "
            f"{tps:.2f} tok/s overall, {steady_tps:.2f} tok/s excluding the first call, "
            f"per-call median {np.median(times):.2f} s"
        )
        log(f"continuation: {text_out!r}")
        np.save(out_dir / f"prompt{i}_ids.npy", np.asarray(ids, np.int32))
        np.save(out_dir / f"prompt{i}_logits.npy", first_logits)
        np.save(out_dir / f"prompt{i}_gen.npy", np.asarray(generated, np.int32))
        summary["prompts"].append(
            {
                "index": i,
                "name": name,
                "kind": kind,
                "text": text,
                "rendered": rendered,
                "prompt_ids": len(ids),
                "top5": [{"id": i_, "token": t, "prob": p} for i_, t, p in top5],
                "generated_ids": generated,
                "generated_text": text_out,
                "stop_reason": reason,
                "seconds_per_call": times,
                "tokens_per_second": tps,
                "tokens_per_second_steady": steady_tps,
            }
        )
        save_summary()
        if time.perf_counter() > deadline:
            log("deadline reached, stopping")
            break
    if args.teacher_force and time.perf_counter() < deadline:
        teacher_force(cfg, tokenizer, runner, log, summary, deadline)
        save_summary()
    log(f"done; oracle in {out_dir}")


def teacher_force(cfg, tokenizer, runner, log, summary, deadline):
    """Mean log-prob of the PARAGRAPH tokens (raw mode, BOS first), one prefill per prefix."""
    ids = [cfg.bos] + list(tokenizer.encode(PARAGRAPH))
    logps, ranks = [], []
    for n in range(1, len(ids)):
        logits, _, _ = runner.step(ids[:n])
        masked = np.asarray(sampling.mask_unused(jnp.asarray(logits), cfg), np.float64)
        z = masked - masked.max()
        logp = z - np.log(np.exp(z).sum())
        logps.append(float(logp[ids[n]]))
        ranks.append(int((masked > masked[ids[n]]).sum()) + 1)
        if time.perf_counter() > deadline:
            break
    logps = np.asarray(logps)
    log(
        f"teacher forcing on {len(logps)} paragraph tokens: mean log-prob {logps.mean():.3f} "
        f"(ppl {np.exp(-logps.mean()):.2f}), top-1 rate {np.mean(np.asarray(ranks) == 1):.2f}, "
        f"min log-prob {logps.min():.2f} at {tokenizer.decode([ids[1 + int(logps.argmin())]])!r}"
    )
    summary["teacher_forcing"] = {
        "paragraph": PARAGRAPH,
        "tokens": len(logps),
        "mean_logprob": float(logps.mean()),
        "perplexity": float(np.exp(-logps.mean())),
        "top1_rate": float(np.mean(np.asarray(ranks) == 1)),
        "logprobs": logps.tolist(),
        "ranks": ranks,
    }


if __name__ == "__main__":
    main()
