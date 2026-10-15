"""Muse Spark TP8/TP16 serving with DSpark or --no-spec single-token decoding."""

import argparse
import collections
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import jax
import jax.numpy as jnp
import numpy as np
import jax.experimental.multihost_utils as mh
from jax.sharding import NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import openai_server
from mpk.chat import (
    CHAT_MODES,
    STOP_IDS,
    MuseTokenStreamer,
    WholeTextStreamer,
    _InteractiveStream,
    muse_split_channels,
    render_chat_tokens,
)
from model_paths import resolve_model_source
from mpk import Config
from mpk import decode_megakernel as model
from mpk import load as mpk_load

DEFAULT_CHECKPOINT = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
DEFAULT_DRAFT = "Inferact/Muse-Spark-Open-DSpark"

PREFILL_TF_MAX = 384
JOB_HEADER = 12
PROMPT_CAPACITY = 8192
STOP_CAPACITY = 2048


def _log(message):
    if jax.process_index() == 0:
        print(message, flush=True)


def _host(value):
    """Local replica of a replicated (P()) value."""
    return np.asarray(value.addressable_shards[0].data)


def _prompt_stream():
    """The launcher's stdin: ``distributed_entry.py`` duplicates fd 0 before JAX
    starts (``K3_STDIN_FD``), so a later redirection of fd 0 cannot cut it off."""
    fd = os.environ.get("K3_STDIN_FD")
    if fd is None:
        return sys.stdin
    try:
        return os.fdopen(int(fd), "r", closefd=False)
    except OSError:
        return sys.stdin


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="HF snapshot (tokenizer, and weights unless --weights)")
    parser.add_argument("--tp", type=int, choices=(8, 16), default=int(os.environ.get("MPK_TP", "8")))
    parser.add_argument("--weights", default=None,
                        help="Packed weight bundle directory (base/ + prefill/ + tiled/ containers; "
                             "default: $MPK_PACKED_WEIGHTS_TP8 or $MPK_PACKED_WEIGHTS_TP16, according to --tp)")
    parser.add_argument("--prompt", action="append",
                        help="Prompt(s) for a non-interactive session; repeatable")
    parser.add_argument("--chat", choices=CHAT_MODES, default="response",
                        help="Chat format: 'response' or 'none' (raw completion)")
    parser.add_argument("--context", type=int, default=4096,
                        help="Attention cache length (multiple of 128)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Generation budget per prompt")
    parser.add_argument("--no-stop", action="store_true", help="Ignore end-of-message tokens")
    parser.add_argument("--ablate", default="", help="Comma-separated kernel ablations (noattn,noxchg,noexpert,nocompute)")
    parser.add_argument("--steps-per-call", type=int, default=4, metavar="N",
                        help="Single-token decode steps riding one device call")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), default=1,
                        help="Maximum independent requests per non-speculative TP8/TP16 batch")
    parser.add_argument("--batch-wait-ms", type=float, default=10.0,
                        help="Maximum queue-coalescing delay for non-speculative batches with --batch-size > 1 (default: 10 ms)")
    speculation = parser.add_mutually_exclusive_group()
    speculation.add_argument("--draft", default=DEFAULT_DRAFT,
                             help="Existing DSpark checkpoint (enabled by default)")
    speculation.add_argument("--no-spec", action="store_const", const=None, dest="draft",
                             help="Use non-speculative decoding; supports batches of 1/2/4/8")
    parser.add_argument("--fixed-acceptance-length", type=int, choices=range(1, 9), default=None,
                        help="Synthetic benchmark only: force accepted drafts plus bonus token")
    parser.add_argument("--verify-backend", choices=("fused", "sequential"), default="fused")
    parser.add_argument("--draft-backend", choices=("fused", "xla"), default="fused")
    parser.add_argument("--dspark-stats", default=None, help="JSONL output for per-request acceptance statistics")
    parser.add_argument("--scan-prefill", action="store_true",
                        help="Use the lax.scan prefill kernel instead of teacher-forced "
                             "decode stepping (requires the prefill/ weight family)")
    parser.add_argument("--serve", default=None, metavar="HOST:PORT",
                        help="Server mode (process 0): /v1/completions and /v1/chat/completions")
    parser.add_argument("--model-name", default=None, help="Model id reported by the server")
    args = parser.parse_args()
    if args.model_name is None:
        mode = "dspark" if args.draft is not None else "nvfp4"
        args.model_name = f"muse-spark-open-{mode}-tp{args.tp}"
    if args.steps_per_call != 4:
        raise ValueError("--steps-per-call must remain 4")
    if args.batch_size > 1 and (args.draft is not None or args.ablate):
        raise ValueError("--batch-size > 1 requires --no-spec without ablations")
    if not np.isfinite(args.batch_wait_ms) or args.batch_wait_ms < 0:
        raise ValueError("--batch-wait-ms must be finite and nonnegative")
    if args.fixed_acceptance_length is not None and args.draft is None:
        raise ValueError("synthetic acceptance requires --draft")
    if args.context % 128:
        raise ValueError("--context must be a multiple of 128")
    checkpoint_revision = (
        os.environ.get("MPK_CHECKPOINT_REVISION")
        if args.checkpoint == DEFAULT_CHECKPOINT
        else None
    )
    args.checkpoint = str(resolve_model_source(args.checkpoint, revision=checkpoint_revision, local_files_only=True))
    if args.weights is None:
        args.weights = os.environ.get(f"MPK_PACKED_WEIGHTS_TP{args.tp}")
        if args.tp == 8 and args.weights is None:
            args.weights = os.environ.get("MPK_PACKED_WEIGHTS")
    if not args.weights:
        parser.error(f"Set MPK_PACKED_WEIGHTS_TP{args.tp} or pass --weights with a pack created by scripts/pack_mpk.sh")

    started = time.perf_counter()

    mesh = model.logical_mesh(args.tp)
    if mesh.size != args.tp:
        raise ValueError(f"TP{args.tp} requires {args.tp} logical devices, found {mesh.size}")
    tp = args.tp

    c = Config.from_checkpoint(args.checkpoint)

    _log(f"[0.0s] loading weights (host {jax.process_index()} places its 8 ranks)")
    base_dev = mpk_load.place_pack(mesh, Path(args.weights) / "base", progress=False, merge_experts=True)
    prefill_dense = mpk_load.place_pack(mesh, Path(args.weights) / "prefill", progress=False)
    decode_dev = model.prepare_decode_weights(mesh, {**base_dev, **prefill_dense})
    prefill_dev = decode_dev
    _log(f"[{time.perf_counter() - started:6.1f}s] weights resident")

    _log(f"[{time.perf_counter() - started:6.1f}s] making programs")
    prefill = model.make_prefill(mesh, c, args.context, tp)
    ablate = frozenset(x for x in args.ablate.split(",") if x)
    decode = model.make_decode(mesh, c, args.context, tp=tp, ablate=ablate)
    _log(f"[{time.perf_counter() - started:6.1f}s] programs made")

    def make_chunk(max_size):
        return model.make_decode_chunk(mesh, c, args.context, max_size, tp=tp, ablate=ablate)

    # One compiled chunk with a dynamic step count: no per-size recompiles and
    # no overshoot beyond what the request needs.
    chunk_dyn = make_chunk(args.steps_per_call)

    def prefill_fn(w, s, toks, n):
        """Teacher-forced prefill: run the decode kernel over the n prompt tokens."""

        def step(i, carry):
            st, _ = carry
            nxt, _, st = decode(
                w, st, jax.lax.dynamic_slice(toks, (i,), (1,)), jnp.array([i], jnp.int32)
            )
            return st, nxt

        return jax.lax.fori_loop(
            0, n, step, (s, jnp.zeros((1,), jnp.int32))
        )

    prefill_chunk = jax.jit(prefill_fn, donate_argnums=(1,))
    prefill_jits = {}
    batch_prefill_jits = {}

    kvh = c.kv_heads // tp

    @jax.jit(out_shardings={"kvcache": NamedSharding(mesh, P("tp"))})
    def zero_states_fn():
        # Device-side fill: no host->device staging per request.
        shape = (tp, c.layers, 1, args.context, kvh * 128)
        z = jnp.zeros(shape, jnp.bfloat16)
        return {"kvcache": z}

    def zero_states_dev():
        return zero_states_fn()

    speculator = None
    if args.draft is not None:
        from mpk.dspark import Decoder

        draft_revision = os.environ.get("MPK_DRAFT_REVISION") if args.draft == DEFAULT_DRAFT else None
        draft_path = resolve_model_source(args.draft, revision=draft_revision, local_files_only=True)
        _log(f"[{time.perf_counter() - started:6.1f}s] loading DSpark from {draft_path}")
        speculator = Decoder(
            mesh, c, args.context, decode_dev, prefill_dev, draft_path, zero_states_dev,
            fixed_length=args.fixed_acceptance_length, backend=args.verify_backend, draft_backend=args.draft_backend,
            stats_path=args.dspark_stats,
        )
        _log(f"[{time.perf_counter() - started:6.1f}s] DSpark resident; {args.verify_backend} verifier, 4 rounds/call")

    @jax.jit
    def first_token(lg):
        """Global argmax over the P('tp')-sharded prefill logits.

        Flattening is rank-major, so the first-max tie-break matches the decode
        kernel's (lowest rank, then lowest in-shard index).
        """
        vals = lg.reshape(lg.shape[0], -1)[:, : c.vocab // tp]
        return jnp.argmax(vals.reshape(-1)).astype(jnp.int32)[None]

    @jax.jit
    def first_tokens(logits):
        values = logits[:, :, :c.vocab // tp].transpose(1, 0, 2)
        return jnp.argmax(values.reshape(values.shape[0], -1), axis=1).astype(jnp.int32)

    def prefill_one(ids):
        if len(ids) + 16 > args.context:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit --context {args.context}")
        toks = np.zeros((args.context,), np.int32)
        toks[: len(ids)] = ids
        states_dev, nxt = prefill_chunk(
            decode_dev,
            zero_states_dev(),
            jnp.asarray(toks),
            jnp.array(len(ids), jnp.int32),
        )
        return states_dev, int(_host(nxt)[0])

    def prefill_one_scan(ids):
        ids = [int(t) for t in ids]
        CHUNK = 64
        width = -(-len(ids) // CHUNK) * CHUNK
        if width + 16 > args.context:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit --context {args.context}")
        npad = width - len(ids)
        padded = [c.pad] * npad + ids
        if width not in prefill_jits:
            prefill_jits[width] = jax.jit(prefill).lower(
                prefill_dev, zero_states_dev(), jnp.zeros((width,), jnp.int32), jnp.array(0, jnp.int32)
            ).compile()
        states_dev, last_logits = prefill_jits[width](
            prefill_dev, zero_states_dev(), jnp.asarray(padded, jnp.int32), jnp.array(npad, jnp.int32)
        )
        return states_dev, int(_host(first_token(last_logits))[0])

    batch_programs = {}

    def get_batch_programs(size):
        if size not in batch_programs:
            batched_decode = model.make_decode_batch(mesh, c, args.context, size)
            batched_chunk = model.make_decode_batch_chunk(mesh, c, args.context, size)

            @jax.jit(out_shardings={"kvcache": NamedSharding(mesh, P("tp"))})
            def zero_batch():
                return {"kvcache": jnp.zeros((tp, c.layers, size, args.context, kvh * 128), jnp.bfloat16)}

            @jax.jit(donate_argnums=(1,))
            def prefill_batch(weights, states, tokens, lengths):
                def step(i, carry):
                    states, previous = carry
                    positions = jnp.where(i < lengths, i, -1)
                    predicted, states = batched_decode(weights, states, tokens[i], positions)
                    return states, jnp.where(i < lengths, predicted, previous)

                return jax.lax.fori_loop(0, jnp.max(lengths), step,
                    (states, jnp.zeros((size,), jnp.int32)))

            batch_programs[size] = zero_batch, prefill_batch, batched_chunk
        return batch_programs[size]

    def generate_batch(specs):
        size = next(n for n in (2, 4, 8) if n >= len(specs))
        zero_batch, prefill_batch, batched_chunk = get_batch_programs(size)
        lengths = np.asarray([len(s[0]) for s in specs] + [0] * (size - len(specs)), np.int32)
        budgets = np.asarray([s[1] for s in specs] + [0] * (size - len(specs)), np.int32)
        started_prefill = time.perf_counter()
        if not args.scan_prefill and max(lengths) <= PREFILL_TF_MAX:
            tokens = np.zeros((PREFILL_TF_MAX, size), np.int32)
            for i, spec in enumerate(specs):
                tokens[:lengths[i], i] = spec[0]
            states, first = prefill_batch(decode_dev, zero_batch(), jnp.asarray(tokens), jnp.asarray(lengths))
            first = _host(first)
        else:
            width = (int(max(lengths)) + 63) // 64 * 64
            if width + 16 > args.context:
                raise ValueError(f"prompt of {max(lengths)} tokens does not fit --context {args.context}")
            pads = width - lengths
            tokens = np.full((size, width), c.pad, np.int32)
            for i, spec in enumerate(specs):
                tokens[i, pads[i]:] = spec[0]
            key = size, width
            if key not in batch_prefill_jits:
                batch_prefill_jits[key] = jax.jit(prefill, donate_argnums=(1,)).lower(
                    prefill_dev, zero_batch(), jnp.asarray(tokens), jnp.asarray(pads)
                ).compile()
            states, logits = batch_prefill_jits[key](prefill_dev, zero_batch(), jnp.asarray(tokens), jnp.asarray(pads))
            first = _host(first_tokens(logits))
        prefill_seconds = time.perf_counter() - started_prefill
        generated = [[] for _ in specs]
        reasons = ["length"] * len(specs)
        finished = np.zeros((size,), bool)
        finished[len(specs):] = True
        decode_seconds = [0.0] * len(specs)
        started_decode = time.perf_counter()

        def finish(i):
            finished[i] = True
            decode_seconds[i] = time.perf_counter() - started_decode
            job = specs[i][4]
            if job is not None:
                job.complete(openai_server.Result(
                    generated=generated[i], finish_reason=reasons[i],
                    prefill_seconds=prefill_seconds, decode_seconds=decode_seconds[i],
                    steps=len(generated[i]),
                ))

        for i, spec in enumerate(specs):
            if budgets[i] > 0 and (first[i] not in STOP_IDS or spec[3]):
                generated[i].append(int(first[i]))
                if spec[4] is not None:
                    spec[4].emit(generated[i])
            elif budgets[i] > 0:
                reasons[i] = "stop"
            if budgets[i] <= 1 or not generated[i]:
                finish(i)

        cur = jnp.asarray(first, jnp.int32)
        positions = lengths.copy()
        remaining = np.maximum(budgets - 1, 0)
        inflight = collections.deque()
        while inflight or np.any((remaining > 0) & ~finished):
            while len(inflight) < 2:
                counts = np.where(finished, 0, np.minimum(4, remaining)).astype(np.int32)
                if not np.any(counts):
                    break
                out = batched_chunk(decode_dev, states, cur, jnp.asarray(np.stack((positions, counts))))
                cur, states = out[:2]
                inflight.append((out, counts))
                positions += counts
                remaining -= counts
            if not inflight:
                break
            out, counts = inflight.popleft()
            tokens = _host(out[2])
            for i, spec in enumerate(specs):
                if finished[i]:
                    continue
                emitted = []
                for token in tokens[:counts[i], i]:
                    generated[i].append(int(token))
                    emitted.append(int(token))
                    if token in STOP_IDS and not spec[3]:
                        reasons[i] = "stop"
                        break
                    if len(generated[i]) >= budgets[i]:
                        break
                if spec[4] is not None:
                    spec[4].emit(emitted)
                if reasons[i] == "stop" or len(generated[i]) >= budgets[i]:
                    finish(i)
        for i in range(len(specs)):
            if not finished[i]:
                finish(i)
        return [(generated[i], reasons[i], prefill_seconds, decode_seconds[i]) for i in range(len(specs))]

    stream_holder = {"current": None}

    def generate(prompt_tokens, max_tokens, stop, ignore_eos=False, job=None):
        """Prefill, then greedily decode to an end token, stop string, or budget.

        Every process runs this in lockstep: outputs are replicated, so all
        hosts see the same tokens and reach the same stop decision. ``job``
        (process 0 only) receives tokens as they are produced.
        """
        ids = [int(t) for t in prompt_tokens]
        if speculator is not None:
            def emit(tokens, dt):
                if job is not None:
                    job.emit(tokens)
                elif stream_holder["current"] is not None and tokens:
                    stream_holder["current"].submit(tokens, dt)
            return speculator.generate(ids, max_tokens, stop, ignore_eos, emit, tokenizer)
        emitted_upto = 0
        last_chunk = {"dt": 0.0}

        def emit_new():
            nonlocal emitted_upto
            if job is not None and len(generated) > emitted_upto:
                job.emit(generated[emitted_upto:])
                emitted_upto = len(generated)
            elif job is None and stream_holder["current"] is not None and len(generated) > emitted_upto:
                stream_holder["current"].submit(generated[emitted_upto:], last_chunk["dt"])
                emitted_upto = len(generated)

        prefill_started = time.perf_counter()
        # Teacher-forced decode stepping wins for short prompts (weight-traffic
        # bound per step); the lax.scan prefill wins for long ones (one pass).
        if args.scan_prefill or len(ids) > PREFILL_TF_MAX:
            states_dev, cur = prefill_one_scan(ids)
        else:
            states_dev, cur = prefill_one(ids)
        prefill_seconds = time.perf_counter() - prefill_started
        generated = [cur] if max_tokens > 0 and (cur not in STOP_IDS or ignore_eos) else []
        finish_reason = "stop" if not generated else "length"
        pos = len(ids)
        decode_seconds = 0.0
        if generated:
            emit_new()
            t0 = time.perf_counter()
            cur_dev = jnp.array([cur], jnp.int32)
            # Async pipeline: keep one chunk in flight ahead of the host readback, so
            # the per-chunk dispatch/emit cost hides behind the device compute. The
            # next chunk's first token flows device-side (chunk output -> input), so
            # the host never blocks the issue path. Chunks past an EOS are trimmed at
            # the host as before (at most one in-flight chunk is wasted).
            inflight = collections.deque()  # (out, step) dispatched but unread
            decoded = 0  # decode steps issued so far
            prev_process_t = time.perf_counter()
            while finish_reason != "stop" and (
                inflight or (decoded < max_tokens - 1 and pos < args.context)
            ):
                while len(inflight) < 2 and decoded < max_tokens - 1 and pos < args.context:
                    step = min(args.steps_per_call, max_tokens - 1 - decoded, args.context - pos)
                    if step <= 0:
                        break
                    out = chunk_dyn(
                        decode_dev, states_dev, cur_dev,
                        np.array([pos, step], np.int32),
                    )
                    cur_dev, states_dev = out[0], out[1]
                    inflight.append((out, step))
                    pos += step
                    decoded += step
                if not inflight:
                    break
                out, step = inflight.popleft()
                tokens = _host(out[2])[:step].tolist()
                now = time.perf_counter()
                last_chunk["dt"] = now - prev_process_t  # apparent chunk period (drives pacing)
                prev_process_t = now
                for t in tokens:
                    if t in STOP_IDS and not ignore_eos:
                        generated.append(int(t))  # the end token is included
                        finish_reason = "stop"
                        break
                    generated.append(int(t))
                    if len(generated) >= max_tokens:
                        break
                emit_new()
                if finish_reason == "stop" or len(generated) >= max_tokens:
                    break
            decode_seconds = time.perf_counter() - t0
        emit_new()
        return generated, finish_reason, prefill_seconds, decode_seconds

    tokenizer = mpk_load.load_tokenizer(args.checkpoint)

    def encode(prompt):
        if args.chat == "none":
            return np.asarray(tokenizer.encode(prompt), np.int32)
        return render_chat_tokens(tokenizer, prompt, args.chat)

    # --- first launches ------------------------------------------------------
    _log(f"[{time.perf_counter() - started:6.1f}s] first launches (compiling)")
    warmup_ids = encode("Hello")
    generated, _, _, _ = generate(warmup_ids, 8, [], job=None)
    for size in (2, 4, 8):
        if size <= args.batch_size:
            generate_batch([(warmup_ids, 8, [], True, None)] * size)
            _log(f"[{time.perf_counter() - started:6.1f}s] batch {size} ready")
    _log(f"[{time.perf_counter() - started:6.1f}s] ready ({tokenizer.decode(generated)!r}...)")

    # --- prompt handling -----------------------------------------------------
    server = None
    if args.serve and jax.process_index() == 0:
        host, _, port = args.serve.rpartition(":")
        server = openai_server.OpenAIServer(
            host or "0.0.0.0", int(port), tokenizer, model_name=args.model_name,
            chat_mode="response",
            default_max_tokens=args.max_tokens,
            max_prompt_tokens=min(PROMPT_CAPACITY, args.context - 2 * 64),
            bos_token_id=None, max_model_len=args.context, log=_log,
            chat_render=lambda tok, messages, mode: render_chat_tokens(tok, messages, mode),
            streamer_factory=lambda tok, chat=False, **kw: (
                MuseTokenStreamer(tok) if chat else WholeTextStreamer(tok)
            ),
            token_channel_splitter=lambda ids: muse_split_channels(tokenizer, ids),
        )
        server.start()
        _log("Ctrl-C ends the server session")

    prompt_stdin = _prompt_stream()

    def read_prompt():
        if args.prompt is not None:
            return args.prompt.pop(0) if args.prompt else None
        while True:
            sys.stdout.write("prompt> ")
            sys.stdout.flush()
            line = prompt_stdin.readline()
            if not line:
                return None
            line = line.rstrip("\n")
            if line.strip().lower() in ("quit", "exit"):
                return None
            if line.strip():
                return line

    def next_jobs():
        """Broadcast one static batch of independent requests to every host."""
        buffer = np.full((args.batch_size, JOB_HEADER + PROMPT_CAPACITY + STOP_CAPACITY), -1, np.int32)
        jobs = []
        if jax.process_index() == 0:
            deadline = None
            while len(jobs) < args.batch_size:
                tokens, max_tokens, stop, ignore_eos = None, args.max_tokens, [], args.no_stop
                job = None
                if server is not None:
                    timeout = 0.0 if jobs else 1.0
                    if deadline is not None:
                        timeout = max(0.0, deadline - time.monotonic())
                    job = server.next_job(timeout=timeout)
                    if job is None:
                        if jobs or server.stopping.is_set():
                            break
                        continue
                    if (speculator is not None or args.batch_size > 1) and (job.temperature != 0 or job.score):
                        job.fail("DSpark and batched TP8/TP16 support greedy generation without prompt log-probabilities")
                        continue
                    budget = args.context - len(job.tokens)
                    stop_bytes = json.dumps(job.stop).encode("utf-8")
                    if budget < 0:
                        job.fail(f"prompt of {len(job.tokens)} tokens does not fit --context {args.context}")
                        continue
                    if len(stop_bytes) > STOP_CAPACITY:
                        job.fail("stop strings too long")
                        continue
                    tokens, max_tokens, stop = job.tokens, min(job.max_tokens, budget), job.stop
                    ignore_eos = job.ignore_eos or args.no_stop
                else:
                    if jobs and args.prompt is None:
                        break
                    prompt = read_prompt()
                    if prompt is None:
                        break
                    tokens = encode(prompt)
                    if len(tokens) + args.max_tokens > args.context:
                        print(f"prompt too long for --context {args.context} with --max-tokens {args.max_tokens}", flush=True)
                        tokens = tokens[: args.context - args.max_tokens]
                if tokens is None:
                    break
                if len(tokens) > PROMPT_CAPACITY:
                    raise ValueError(f"prompt exceeds the {PROMPT_CAPACITY}-token broadcast capacity")
                row = buffer[len(jobs)]
                stop_bytes = json.dumps(stop).encode("utf-8")
                row[:5] = (1, len(tokens), max_tokens, len(stop_bytes), int(ignore_eos))
                row[JOB_HEADER:JOB_HEADER + len(tokens)] = tokens
                row[JOB_HEADER + PROMPT_CAPACITY:JOB_HEADER + PROMPT_CAPACITY + len(stop_bytes)] = np.frombuffer(stop_bytes, np.uint8)
                jobs.append(job)
                if len(jobs) == 1 and args.batch_size > 1 and speculator is None:
                    deadline = time.monotonic() + args.batch_wait_ms / 1000
        buffer = np.asarray(mh.broadcast_one_to_all(buffer))
        specs = []
        for i, row in enumerate(buffer):
            if row[0] != 1:
                break
            count, max_tokens, stop_length, ignore_eos = (int(v) for v in row[1:5])
            tokens = row[JOB_HEADER:JOB_HEADER + count]
            stop_bytes = row[JOB_HEADER + PROMPT_CAPACITY:JOB_HEADER + PROMPT_CAPACITY + stop_length].astype(np.uint8).tobytes()
            specs.append((tokens, max_tokens, json.loads(stop_bytes.decode("utf-8")) if stop_length else [],
                          bool(ignore_eos), jobs[i] if jax.process_index() == 0 else None))
        return specs

    interactive = server is None and args.prompt is None and jax.process_index() == 0
    while True:
        job_specs = next_jobs()
        if not job_specs:
            break
        if len(job_specs) > 1:
            try:
                results = generate_batch(job_specs)
            except Exception as error:
                for spec in job_specs:
                    if spec[4] is not None and not spec[4].done.is_set():
                        spec[4].fail(f"{type(error).__name__}: {error}")
                raise
            for spec, result in zip(job_specs, results):
                generated, finish_reason, _, _ = result
                if spec[4] is None and jax.process_index() == 0:
                    print(tokenizer.decode(generated), flush=True)
                _log(f"batch {len(job_specs)}: {len(spec[0])} prompt tokens, {len(generated)} generated, finish {finish_reason}")
            continue
        prompt_tokens, max_tokens, stop, ignore_eos, job = job_specs[0]
        stream = None
        if job is None and jax.process_index() == 0:
            stream = _InteractiveStream(tokenizer, chat=args.chat != "none")
            stream_holder["current"] = stream
            print(flush=True)
        try:
            generated, finish_reason, prefill_seconds, decode_seconds = generate(
                prompt_tokens, max_tokens, stop, ignore_eos, job
            )
        except Exception as error:
            if stream is not None:
                stream_holder["current"] = None
                stream.close()
            if job is not None:
                job.fail(f"{type(error).__name__}: {error}")
            raise
        if stream is not None:
            stream_holder["current"] = None
            stream.close()
            print(flush=True)
        if job is not None:
            job.complete(openai_server.Result(
                generated=generated, finish_reason=finish_reason,
                prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
                steps=len(generated),
            ))
            _log(f"{job.kind}: {len(prompt_tokens)} prompt tokens, {len(generated)} generated in "
                 f"{decode_seconds:.2f} s, finish {finish_reason}")
        elif jax.process_index() == 0:
            tokens_out = max(len(generated), 1)
            print(f"--- {len(generated)} tokens in {decode_seconds:.2f} s: "
                  f"{tokens_out / max(decode_seconds, 1e-9):.1f} tokens/s", flush=True)
        if interactive:
            print(flush=True)
    if server is not None:
        server.shutdown()
    _log("bye")


if __name__ == "__main__":
    main()
