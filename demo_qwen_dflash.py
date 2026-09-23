"""Interactive Qwen3.8-27B + DFlash2 speculative decoding demo on the TP8 megakernel.

Loads the Qwen3.8 checkpoint (packed weight containers or the HF snapshot) and
the DFlash2 draft (fused Pallas draft kernel), then reads prompts from the
terminal and prints responses as the speculative loop runs. After each
response it prints latency and acceptance metrics, and with ``--baseline``
also decodes the same prompt greedily one token at a time for comparison.

Single host: one JAX process drives all eight devices of four TPU chips.
Run under the launcher (``scripts/demo_qwen_dflash.sh``); pass ``--prompt``
one or more times for a non-interactive session, or ``--serve HOST:PORT`` for
the OpenAI-compatible endpoint.
"""

import argparse
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import openai_server
from model_paths import resolve_model_source
from qwen import Config
from qwen import decode_megakernel as model
from qwen import dflash
from qwen import load as qwen_load

DEFAULT_CHECKPOINT = "Qwen/Qwen3.8-27B"
DEFAULT_DRAFT = "z-lab/Qwen3.8-27B-DFlash2"

CHAT_MODES = ("none", "response", "think")
STOP_IDS = frozenset((248044, 248046))  # <|endoftext|>, <|im_end|>
CHUNK = 64  # prefill width granularity


def render_chat_tokens(tokenizer, messages, mode):
    """Qwen3.8 chat: system/user/assistant turns in im_start/im_end segments.

    ``mode == "response"`` closes an empty think block at the assistant turn
    (thinking disabled); ``mode == "think"`` leaves the think block open. The
    assistant generation prefix ends the sequence. ``messages`` follows the
    OpenAI shape; a bare string is treated as one user turn.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    scaffold = "<think>\n\n</think>\n\n"  # response mode: thinking off
    text = []
    for message in messages:
        role = message.get("role", "user")
        body = message.get("content") or ""
        if isinstance(body, list):  # OpenAI content parts; text parts only
            body = "".join(
                part.get("text", "") for part in body
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if role in ("system", "user"):
            text.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
        elif role == "assistant":
            text.append(f"<|im_start|>assistant\n{scaffold}{body}<|im_end|>\n")
        else:
            raise ValueError(f"unsupported message role {role!r}")
    text.append("<|im_start|>assistant\n")
    if mode == "think":
        text.append("<think>\n")
    else:
        text.append(scaffold)
    return np.asarray(tokenizer.encode("".join(text)), np.int32)


def split_think_channels(text):
    """Split a decoded Qwen turn into reasoning and final-answer content."""
    if "</think>" not in text:
        return None, text
    reasoning, content = text.split("</think>", 1)
    return reasoning.strip(), content.strip()


class WholeTextStreamer:
    """Emits text deltas by re-decoding the full token list on each feed.

    The fused decode loop emits tokens in chunks; re-decoding keeps multi-byte
    characters and BPE merges intact. End tokens are never emitted.
    """

    def __init__(self, tokenizer, **_unused):
        self.tokenizer = tokenizer
        self.tokens = []
        self.text = ""

    def feed(self, tokens):
        self.tokens.extend(int(t) for t in tokens if int(t) not in STOP_IDS)
        new = self.tokenizer.decode(self.tokens)
        delta = new[len(self.text) :]
        self.text = new
        return [("content", delta)] if delta else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="HF snapshot (tokenizer, and weights unless --weights)")
    parser.add_argument("--weights", default=None,
                        help="Packed weight directory (with untiled/ and tiled/ containers; "
                             "default: pack from --checkpoint)")
    parser.add_argument("--draft", default=DEFAULT_DRAFT, help="DFlash2 draft HF snapshot")
    parser.add_argument("--draft-weights", default=None,
                        help="Packed draft kernel directory (with the kernel/ container)")
    parser.add_argument("--pack-to", default=None, metavar="DIR",
                        help="After packing from the checkpoints, write the untiled/, tiled/ and "
                             "kernel/ containers under DIR (then: --weights DIR --draft-weights DIR)")
    parser.add_argument("--prompt", action="append", help="Prompt(s) for a non-interactive session; repeatable")
    parser.add_argument("--chat", choices=CHAT_MODES, default="response",
                        help="Chat format: 'response' (no thinking), 'think', or 'none' (raw completion)")
    parser.add_argument("--context", type=int, default=2048, help="Attention cache length (multiple of 256)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Generation budget per prompt")
    parser.add_argument("--baseline", action="store_true", help="Also decode greedily one token at a time")
    parser.add_argument("--no-spec", action="store_true", help="Decode with the plain single-token kernel")
    parser.add_argument("--no-stop", action="store_true", help="Ignore end-of-message tokens")
    parser.add_argument("--steps-per-call", type=int, default=256, metavar="N",
                        help="Steps riding one device call: N single-token decode steps per call "
                             "in the plain loop, or N speculation rounds per call in --force-accept "
                             "mode (N x acceptance tokens stream per server event). One call covers "
                             "the whole request once N x tokens-per-step reaches the budget")
    parser.add_argument("--force-accept", type=int, default=None, metavar="N",
                        help="Synthetic-acceptance bench mode: accept exactly N of the 7 drafted "
                             "tokens every round (bypasses real acceptance; for perf measurement only). "
                             "Stop tokens are disabled")
    parser.add_argument("--serve", default=None, metavar="HOST:PORT",
                        help="Server mode: answer OpenAI-compatible /v1/completions and /v1/chat/completions "
                             "requests (greedy; one request at a time) instead of reading prompts")
    parser.add_argument("--model-name", default="qwen3.8-27b-dflash2", help="Model id reported by the server")
    args = parser.parse_args()
    if args.context % 256 or args.context < 512:
        raise ValueError("--context must be a multiple of 256 and at least 512")
    checkpoint_revision = (
        os.environ.get("QWEN_CHECKPOINT_REVISION")
        if args.checkpoint == DEFAULT_CHECKPOINT
        else None
    )
    draft_revision = (
        os.environ.get("QWEN_DRAFT_REVISION") if args.draft == DEFAULT_DRAFT else None
    )
    args.checkpoint = str(resolve_model_source(args.checkpoint, revision=checkpoint_revision))
    args.draft = str(resolve_model_source(args.draft, revision=draft_revision))
    packed = Path(os.environ.get("QWEN_PACKED_WEIGHTS", ROOT / "checkpoints/qwen38-tp8"))
    if args.weights is None and (packed / "untiled" / "manifest.json").exists():
        args.weights = str(packed)
    if args.draft_weights is None and (packed / "kernel" / "manifest.json").exists():
        args.draft_weights = str(packed)

    tp = 8
    devices = jax.devices()
    if len(devices) < tp:
        raise RuntimeError(f"need {tp} TPU devices, found {len(devices)} (the launcher exposes four chips)")
    mesh = jax.sharding.Mesh(np.asarray(devices[:tp]), ("tp",))
    replicated = NamedSharding(mesh, P())

    c = Config.from_checkpoint(args.checkpoint)
    dparams, dcfg = dflash.load_draft(args.draft)
    B = dcfg.block_size

    log = lambda m: print(f"[{time.perf_counter() - started:6.1f}s] {m}", flush=True)
    started = time.perf_counter()

    log("loading target weights")
    untiled, tiled = qwen_load.load_target(args.checkpoint, c, tp, packed=args.weights)
    if args.pack_to:
        # keep the packed containers for fast future startups
        qwen_load.save_pack(Path(args.pack_to) / "untiled", untiled)
        qwen_load.save_pack(Path(args.pack_to) / "tiled", tiled)
    packed_untiled = qwen_load.build_sharded(mesh, untiled)
    packed_tiled = qwen_load.build_sharded(mesh, tiled)
    del untiled, tiled
    log("target weights resident")

    if args.draft_weights and (Path(args.draft_weights) / "kernel" / "manifest.json").exists():
        dk_np = qwen_load.load_pack(Path(args.draft_weights) / "kernel")
    else:
        dk_np = dflash.pack_draft_kernel(dparams, dcfg, tp)
    if args.pack_to:
        qwen_load.save_pack(Path(args.pack_to) / "kernel", jax.tree.map(np.asarray, dk_np))
        log(f"packed containers written to {args.pack_to}")
    dk = qwen_load.build_sharded(mesh, jax.tree.map(jnp.asarray, dk_np))
    draft_dev = qwen_load.build_sharded(mesh, dflash.pack_draft_tp(dparams, dcfg, tp))
    # the fused draft kernel works in a permuted channel space; the shared
    # embedding / LM head / fc weights are fed as permuted copies
    perm = dk_np["perm"]
    emb_p = qwen_load.build_sharded(mesh, np.take(np.asarray(packed_untiled["emb"]), perm, axis=-1))
    lmw_p = qwen_load.build_sharded(mesh, np.take(np.asarray(packed_untiled["lmw"]), perm, axis=1))
    fc_p = qwen_load.build_sharded(mesh, np.take(np.asarray(draft_dev["fc"]), perm, axis=1))
    hnorm_p = qwen_load.build_sharded(mesh, np.take(np.asarray(draft_dev["hidden_norm"]), perm, axis=-1))
    predcb = jax.device_put(jnp.asarray(dparams["selector"]["predecessor_codebook"]), replicated)
    succcb = jax.device_put(jnp.asarray(dparams["selector"]["successor_codebook"]), replicated)
    log("draft weights resident")

    # --- programs ------------------------------------------------------------
    prefill = model.make_prefill(mesh, c, args.context, tp, tap_layers=dcfg.target_layer_ids)
    decode = model.make_decode(mesh, c, args.context, tp=tp)
    verify = model.make_verify_block(mesh, c, args.context, tp, dcfg.target_layer_ids, block=B)
    draft_k = dflash.make_draft_kernel(mesh, dcfg, tp, block=B, context=args.context)
    ctx_update = dflash.make_ctx_update(mesh, dcfg, len(dcfg.target_layer_ids) * c.dim, tp)
    scan_fn = dflash.make_scan_fn(
        draft_k, verify, c, dcfg, args.context,
        stop_ids=(-1, -1) if (args.no_stop or args.force_accept is not None) else tuple(sorted(STOP_IDS)),
        force_acc=args.force_accept,
    )

    def make_chunk(size):
        def fn(w, s, tok_, pos):
            def step(i, carry):
                cur, st, out = carry
                nxt, _, st = decode(w, st, cur, pos + jnp.array([i, 0], jnp.int32))
                return nxt, st, out.at[i].set(nxt[0])

            return jax.lax.fori_loop(0, size, step, (tok_, s, jnp.zeros((size,), jnp.int32)))

        return jax.jit(fn, donate_argnums=(1,))

    chunk_plain = make_chunk(args.steps_per_call)
    prefill_jits = {}

    def zero_states_dev():
        s0 = qwen_load.pack_states(qwen_load.zero_states(c, args.context), c, tp)
        return qwen_load.build_sharded(mesh, jax.tree.map(jnp.asarray, s0))

    def zero_snaps_dev():
        nl = c.layers - len(c.full_attention)
        vh_r, kh_r = c.v_heads // tp, c.k_heads // tp
        qkvw_r = 2 * kh_r * c.k_dim + vh_r * c.v_dim
        return qwen_load.build_sharded(
            mesh,
            {
                "conv": jnp.zeros((tp, nl, B * c.conv_size, qkvw_r), jnp.bfloat16),
                "rec": jnp.zeros((tp, nl, B, vh_r, c.v_dim, c.k_dim), jnp.float32),
            },
        )

    def prefill_one(ids):
        width = -(-len(ids) // CHUNK) * CHUNK
        if width + 16 > args.context:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit --context {args.context}")
        npad = width - len(ids)
        padded = [0] * npad + ids
        if width not in prefill_jits:
            prefill_jits[width] = jax.jit(prefill).lower(
                packed_untiled, zero_states_dev(), jnp.zeros((width,), jnp.int32), jnp.array(0, jnp.int32)
            ).compile()
        return prefill_jits[width](
            packed_untiled, zero_states_dev(), jnp.asarray(padded, jnp.int32), jnp.array(npad, jnp.int32)
        ), npad

    def generate(prompt_tokens, max_tokens, stop, ignore_eos=False, job=None):
        """Prefill, then decode to an end token, a stop string, or ``max_tokens``.

        Returns (generated ids, finish reason, prefill seconds, decode seconds,
        speculative rounds, produced tokens). The fused speculative scan runs
        the whole generation in one device-side loop, so ``job`` receives the
        new tokens once, at completion.
        """
        ids = [int(t) for t in prompt_tokens]
        emitted_upto = 0

        def emit_new():
            nonlocal emitted_upto
            if job is not None and len(generated) > emitted_upto:
                job.emit(generated[emitted_upto:])
                emitted_upto = len(generated)

        prefill_started = time.perf_counter()
        (states_dev, last_logits, prompt_feats), npad = prefill_one(ids)
        prefill_seconds = time.perf_counter() - prefill_started
        cur = int(
            np.argmax(
                np.concatenate(
                    [np.asarray(last_logits[r])[: c.vocab // tp] for r in range(tp)]
                )
            )
        )
        generated = [cur] if max_tokens > 0 and (cur not in STOP_IDS or ignore_eos) else []
        finish_reason = "stop" if not generated else "length"
        pos = npad + len(ids)
        decode_seconds = 0.0
        rounds = produced = 0
        if generated:
            if args.no_spec:
                emit_new()  # the first token right after prefill
                t0 = time.perf_counter()
                while len(generated) < max_tokens and pos + args.steps_per_call <= args.context:
                    nxt, states_dev, out = chunk_plain(
                        packed_tiled, states_dev, jnp.array([cur], jnp.int32),
                        jnp.array([pos, npad], jnp.int32),
                    )
                    pos += args.steps_per_call
                    rounds += 1
                    for t in np.asarray(out).tolist():
                        if t in STOP_IDS and not ignore_eos:
                            generated.append(int(t))  # the end token is included
                            finish_reason = "stop"
                            break
                        generated.append(int(t))
                        cur = int(t)
                        if len(generated) >= max_tokens:
                            break
                    emit_new()
                    if finish_reason == "stop" or len(generated) >= max_tokens:
                        break
                decode_seconds = time.perf_counter() - t0
            else:
                featbuf = jax.device_put(
                    jnp.zeros((args.context, len(dcfg.target_layer_ids) * c.dim), jnp.bfloat16), replicated
                )
                featbuf = jax.lax.dynamic_update_slice(featbuf, prompt_feats, (0, 0))
                # 16 trailing dummy rows: the kernel's folded ctx update writes
                # there on the first round
                ctxbuf = jax.device_put(jnp.zeros((args.context + 16, dcfg.hidden), jnp.bfloat16), replicated)
                ctxbuf = ctx_update(fc_p, hnorm_p, featbuf, ctxbuf, jnp.array(0, jnp.int32))
                snaps_dev = zero_snaps_dev()
                prefill_seconds = time.perf_counter() - prefill_started  # setup rides in TTFT
                emit_new()  # the first token; the client TPOT window is the scan below
                t0 = time.perf_counter()
                if args.force_accept is None:
                    limit = jnp.array(min(pos + max_tokens, args.context - B), jnp.int32)
                    tokens, start_f, rounds_, produced_, states_dev, _, _, _ = scan_fn(
                        dk, lmw_p, emb_p, predcb, succcb, packed_tiled, states_dev, snaps_dev,
                        featbuf, ctxbuf, jnp.array(cur, jnp.int32), jnp.array(pos, jnp.int32),
                        jnp.array(npad, jnp.int32), limit,
                    )
                    rounds, produced = int(rounds_), int(produced_)
                    stream = np.asarray(tokens)[pos + 1 : int(start_f) + 1].tolist()
                    for t in stream:
                        if t in STOP_IDS and not ignore_eos:
                            generated.append(int(t))  # the end token is included
                            finish_reason = "stop"
                            break
                        generated.append(int(t))
                    generated = generated[:max_tokens]
                else:
                    # bench mode: chunked scan calls so a streaming client sees
                    # tokens incrementally (the round time is content-independent)
                    # each call runs steps-per-call rounds; with forced acceptance that is
                    # exactly steps-per-call x acceptance tokens
                    per_call = args.steps_per_call * (args.force_accept + 1)
                    target = min(pos + max_tokens, args.context - B)
                    while pos < target and len(generated) < max_tokens:
                        limit = jnp.array(min(pos + per_call, target), jnp.int32)
                        tokens, start_f, rounds_, produced_, states_dev, snaps_dev, featbuf, ctxbuf = scan_fn(
                            dk, lmw_p, emb_p, predcb, succcb, packed_tiled, states_dev, snaps_dev,
                            featbuf, ctxbuf, jnp.array(cur, jnp.int32), jnp.array(pos, jnp.int32),
                            jnp.array(npad, jnp.int32), limit,
                        )
                        rounds += int(rounds_)
                        produced += int(produced_)
                        new_pos = int(start_f)
                        generated.extend(int(t) for t in np.asarray(tokens)[pos + 1 : new_pos + 1])
                        cur = int(tokens[new_pos])
                        pos = new_pos
                        emit_new()
                    generated = generated[:max_tokens]
                decode_seconds = time.perf_counter() - t0
        emit_new()
        return generated, finish_reason, prefill_seconds, decode_seconds, rounds, produced

    tokenizer = qwen_load.load_tokenizer(args.checkpoint)

    def encode(prompt):
        if args.chat == "none":
            return np.asarray(tokenizer.encode(prompt), np.int32)
        return render_chat_tokens(tokenizer, prompt, args.chat)

    # --- first launches ------------------------------------------------------
    log("first launches (compiling)")
    warmup_ids = encode("Hello")
    generated, _, _, _, _, _ = generate(warmup_ids, 8, [], job=None)
    log(f"ready ({tokenizer.decode(generated)!r}...)")

    # --- prompt handling -----------------------------------------------------
    server = None
    if args.serve:
        host, _, port = args.serve.rpartition(":")
        server = openai_server.OpenAIServer(
            host or "0.0.0.0", int(port), tokenizer, model_name=args.model_name,
            chat_mode="think" if args.chat == "think" else "response",
            default_max_tokens=args.max_tokens,
            max_prompt_tokens=args.context - 2 * CHUNK,
            bos_token_id=None, max_model_len=args.context, log=log,
            chat_render=lambda tok, messages, mode: render_chat_tokens(tok, messages, mode),
            streamer_factory=lambda tok, **kw: WholeTextStreamer(tok),
            channel_splitter=split_think_channels,
        )
        server.start()
        log("Ctrl-C ends the server session")

    def read_prompt():
        if args.prompt is not None:
            return args.prompt.pop(0) if args.prompt else None
        while True:
            sys.stdout.write("prompt> ")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                return None
            line = line.rstrip("\n")
            if line.strip().lower() in ("quit", "exit"):
                return None
            if line.strip():
                return line

    def next_job():
        """(tokens, max_tokens, stop, ignore_eos, job) or None to end the session."""
        if server is not None:
            while True:
                job = server.next_job(timeout=1.0)
                if job is None:
                    continue
                budget = args.context - len(job.tokens) - B
                if budget < 0:
                    job.fail(f"prompt of {len(job.tokens)} tokens does not fit --context {args.context}")
                    continue
                return job.tokens, min(job.max_tokens, budget), job.stop, job.ignore_eos or args.no_stop, job
        prompt = read_prompt()
        if prompt is None:
            return None
        tokens = encode(prompt)
        if len(tokens) + args.max_tokens + B > args.context:
            print(f"prompt too long for --context {args.context} with --max-tokens {args.max_tokens}", flush=True)
            tokens = tokens[: args.context - args.max_tokens - B]
        return tokens, args.max_tokens, [], args.no_stop, None

    interactive = server is None and args.prompt is None
    while True:
        job_spec = next_job()
        if job_spec is None:
            break
        prompt_tokens, max_tokens, stop, ignore_eos, job = job_spec
        try:
            generated, finish_reason, prefill_seconds, decode_seconds, rounds, produced = generate(
                prompt_tokens, max_tokens, stop, ignore_eos, job
            )
        except Exception as error:  # report to the client, keep the session behavior aligned
            if job is not None:
                job.fail(f"{type(error).__name__}: {error}")
            raise
        if job is not None:
            job.complete(openai_server.Result(
                generated=generated, finish_reason=finish_reason,
                prefill_seconds=prefill_seconds, decode_seconds=decode_seconds, steps=rounds,
            ))
            log(f"{job.kind}: {len(prompt_tokens)} prompt tokens, {len(generated)} generated in "
                f"{decode_seconds:.2f} s ({rounds} rounds), finish {finish_reason}")
        else:
            visible = generated[:-1] if finish_reason == "stop" and generated and generated[-1] in STOP_IDS else generated
            print(f"\n{tokenizer.decode(visible)}\n", flush=True)
            tokens_out = max(len(generated), 1)
            spec_note = (f"; {rounds} speculative rounds, {produced / max(rounds, 1):.2f} tokens/round"
                         if not args.no_spec else "")
            print(f"--- {len(generated)} tokens in {decode_seconds:.2f} s: "
                  f"{len(generated) / max(decode_seconds, 1e-9):.1f} tokens/s{spec_note}", flush=True)
        if args.baseline and server is None:
            base_started = time.perf_counter()
            (states_b, last_logits_b, _), npad_b = prefill_one([int(t) for t in prompt_tokens])
            cur_b = int(np.argmax(np.concatenate(
                [np.asarray(last_logits_b[r])[: c.vocab // tp] for r in range(tp)])))
            base = [cur_b]
            pos_b = npad_b + len(prompt_tokens)
            while len(base) < len(generated) and pos_b + args.steps_per_call <= args.context:
                nxt, states_b, out = chunk_plain(
                    packed_tiled, states_b, jnp.array([cur_b], jnp.int32), jnp.array([pos_b, npad_b], jnp.int32)
                )
                pos_b += args.steps_per_call
                for t in np.asarray(out).tolist():
                    base.append(int(t))
                    cur_b = int(t)
                    if len(base) >= len(generated):
                        break
            base_seconds = time.perf_counter() - base_started
            base = base[: len(generated)]
            agree = 0
            while agree < min(len(base), len(generated)) and base[agree] == generated[agree]:
                agree += 1
            print(f"    single-token baseline: {len(base)} tokens in {base_seconds:.2f} s; "
                  f"speedup {base_seconds / max(decode_seconds, 1e-9):.2f}x; agrees on the first {agree} tokens",
                  flush=True)
        if interactive:
            print(flush=True)
    if server is not None:
        server.shutdown()
    log("bye")


if __name__ == "__main__":
    main()
