"""Muse Spark 1.2 (816B-A42B) on the TP8 decode megakernel: interactive demo, server, benchmark.

Loads the pre-sharded int4 container (`musespark.load.load_presharded`), builds the XLA prefill
(`musespark.prefill`) and the decode megakernel program (`musespark.decode_megakernel`), then
reads prompts from the terminal (chat-rendered with `musespark.chat`) and streams the answers.
`--prompt` (repeatable) runs non-interactively, `--serve HOST:PORT` answers OpenAI-compatible
requests (`openai_server.py`), `--bench` measures decode steps for B in {1, 2, 4, 8}.

Generation: every prompt is prefilled into its own KV-cache row (one XLA program per padded
prompt length), the batch then runs `--steps-per-call` decode steps per device call inside a
`lax.fori_loop`; stop ids (`<|end_of_text|>`, `<|eot|>`) and budgets are handled on the host
after each call. Tokens are greedy (`--greedy`) or sampled with temperature / top-k / top-p
(Muse Spark defaults 1.0 / 64 / 1.0) from the full logits.

Single host, one process driving eight devices of four chips: run through the launcher
`scripts/demo_musespark.sh` (see it for the environment; while other agents hold chip locks,
prefix the launcher with `tpu_run.sh all`).
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import openai_server
from musespark import chat, sampling
from musespark import load as ms_load
from musespark.prefill import CHUNK, make_prefill, pad_prompt

DEFAULT_WEIGHTS = "/filestore/weights/muse-spark-tp8-int4"
DEFAULT_CHECKPOINT = "/filestore/weights/Muse-Spark-1.2-816B-A42B-open"
BENCH_BATCHES = (1, 2, 4, 8)
TP = 8


# --- decode program call site (adapt here if `make_decode` changes) -----------------------------
def build_decode(mesh, cfg, context, batch, greedy):
    """`decode(weights, caches, tokens [B] i32, pos [B] i32) -> (next_tokens [B], logits, caches)`.

    design.md 5.6: greedy programs return the argmax tokens (logits may be None); sampling
    programs are built with `return_logits=True` and must return the softcapped full-vocabulary
    logits `[B, V]` (unused ids masked or not; `sampling.sample` never picks masked ids and the
    demo masks them itself).
    """
    from musespark import decode_megakernel

    return decode_megakernel.make_decode(
        mesh, cfg, context, batch, greedy=greedy, return_logits=not greedy
    )


def make_chunk(decode, steps, cfg, greedy, top_k):
    """`steps` decode steps per device call (Qwen `make_chunk` idiom).

    `(weights, caches, tokens [B], pos [B], key, temperature, top_p) -> (tokens, caches, out
    [steps, B], key)`: row b's token at `pos[b] + i` is `out[i, b]`; sampling draws a fresh
    subkey per step so a call is a deterministic function of `key`.
    """

    def fn(weights, caches, tokens, pos, key, temperature, top_p):
        def step(i, carry):
            cur, caches, out, key = carry
            nxt, logits, caches = decode(weights, caches, cur, pos + i)
            if not greedy:
                key, sub = jax.random.split(key)
                nxt = sampling.sample(
                    sampling.mask_unused(logits, cfg), sub, temperature, top_k, top_p
                )
            return nxt.astype(jnp.int32), caches, out.at[i].set(nxt), key

        out = jnp.zeros((steps, tokens.shape[0]), jnp.int32)
        return lax.fori_loop(0, steps, step, (tokens, caches, out, key))

    return jax.jit(fn, donate_argnums=(1,))


# --- text streaming ------------------------------------------------------------------------------
class MuseStreamer:
    """Text deltas per channel from generated tokens (re-decodes the whole turn on each feed).

    The generation prompt ends with `<|start|>assistant`; the model continues with
    ` to=self<|message|>...<|eom|>` (reasoning) and/or ` to=user<|message|>...<|eot|>` (content),
    which `chat.parse_assistant` separates. Stop tokens are never emitted as text.
    """

    def __init__(self, tokenizer, **_unused):
        self.tokenizer = tokenizer
        self.tokens = []
        self.emitted = {"reasoning": "", "content": ""}

    def feed(self, tokens):
        self.tokens.extend(int(t) for t in tokens if int(t) not in chat.STOP_IDS)
        reasoning, content = chat.parse_assistant("assistant" + self.tokenizer.decode(self.tokens))
        deltas = []
        for channel, text in (("reasoning", reasoning), ("content", content)):
            text = text.removesuffix("\ufffd")  # an incomplete multi-byte character: wait
            delta = text[len(self.emitted[channel]) :]
            if text.startswith(self.emitted[channel]) and delta:
                deltas.append((channel, delta))
                self.emitted[channel] = text
        return deltas


def split_channels(text):
    """(reasoning or None, content) of a decoded assistant turn."""
    reasoning, content = chat.parse_assistant("assistant" + text)
    if not reasoning and not content:  # no message header at all: raw completion text
        return None, text
    return (reasoning or None), content


def terminal_printer(streamer):
    """`on_tokens(row, tokens)` callback that streams a turn to stdout, channel switches marked."""
    channel = [None]

    def on_tokens(_row, tokens):
        for name, text in streamer.feed(tokens):
            if name != channel[0]:
                sys.stdout.write(f"\n[{name}] ")
                channel[0] = name
            sys.stdout.write(text)
        sys.stdout.flush()

    return on_tokens


# --- session -------------------------------------------------------------------------------------
class Session:
    """Weights, caches and compiled programs for one decode batch size."""

    def __init__(self, mesh, cfg, weights, args, batch, log):
        self.mesh, self.cfg, self.weights, self.args, self.batch, self.log = (
            mesh,
            cfg,
            weights,
            args,
            batch,
            log,
        )
        self.context = args.context
        self.greedy = args.greedy
        self.caches = ms_load.zero_caches(mesh, cfg, batch, args.context)
        self.prefill_program = make_prefill(mesh, cfg, args.context, TP)
        self.prefill_buckets = set()  # padded lengths already compiled
        self.decode = build_decode(mesh, cfg, args.context, batch, args.greedy)
        self.chunk = make_chunk(self.decode, args.steps_per_call, cfg, args.greedy, args.top_k)
        self.replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        self.key = self.replicate_key(jax.random.key(args.seed))
        self.first_sampler = jax.jit(
            lambda logits, key, t, p: sampling.sample(
                sampling.mask_unused(logits, cfg)[None], key, t, args.top_k, p
            )[0]
        )
        self.first_greedy = jax.jit(lambda logits: jnp.argmax(sampling.mask_unused(logits, cfg)))

    def replicate(self, x):
        """Small int32 array replicated on the mesh (the sharding the decode program returns):
        jit specialises on input shardings, so a plain single-device `jnp.asarray` for the
        first call and the program's own replicated outputs for the later ones would compile
        the chunk program twice."""
        return jax.device_put(jnp.asarray(x, jnp.int32), self.replicated)

    def replicate_key(self, key):
        """The chunk program returns its key replicated on the mesh; a fresh single-device key
        would compile the program a second time (see `replicate`)."""
        return jax.device_put(key, self.replicated)

    def next_key(self):
        self.key, sub = jax.random.split(self.key)
        return sub

    def prefill(self, ids, row):
        """Prefill `ids` into cache row `row`; returns (softcapped last logits [V], seconds).

        One executable per padded length: the first call of a bucket compiles (logged).
        """
        tokens, length = pad_prompt(self.cfg, ids)
        bucket = tokens.shape[0]
        if bucket + self.args.steps_per_call > self.context:
            raise ValueError(f"prompt of {length} tokens does not fit --context {self.context}")
        compiling = bucket not in self.prefill_buckets
        started = time.perf_counter()
        logits, self.caches = self.prefill_program(
            self.weights, self.caches, tokens, jnp.int32(length), jnp.int32(row)
        )
        logits.block_until_ready()
        seconds = time.perf_counter() - started
        if compiling:
            self.prefill_buckets.add(bucket)
            self.log(f"prefill program for {bucket} tokens: compile + first run {seconds:.1f} s")
        return logits, seconds

    def generate(
        self,
        prompts,
        max_tokens,
        ignore_eos=False,
        temperature=None,
        top_p=None,
        seed=None,
        on_tokens=None,
    ):
        """Generate for up to `batch` prompts (lists of ids) at once.

        Returns (generated per prompt (stop token included when one ended it), finish reasons,
        prefill seconds, decode seconds, decode steps). `on_tokens(row, new_tokens)` is called
        after every device call with the row's newly generated tokens.
        """
        if len(prompts) > self.batch:
            raise ValueError(f"{len(prompts)} prompts for a batch of {self.batch}")
        temperature = self.args.temperature if temperature is None else temperature
        top_p = self.args.top_p if top_p is None else top_p
        if seed is not None:
            self.key = self.replicate_key(jax.random.key(seed))
        cfg = self.cfg
        n = len(prompts)
        first = np.full((self.batch,), cfg.pad, np.int32)
        pos = np.zeros((self.batch,), np.int32)
        prefill_seconds = 0.0
        for b, ids in enumerate(prompts):
            logits, seconds = self.prefill(ids, b)
            prefill_seconds += seconds
            if self.greedy:
                first[b] = int(self.first_greedy(logits))
            else:
                first[b] = int(self.first_sampler(logits, self.next_key(), temperature, top_p))
            pos[b] = len(ids)
        generated = [[] for _ in range(n)]
        finished = [False] * n
        reasons = ["length"] * n

        def push(b, token):
            if finished[b]:
                return
            generated[b].append(int(token))
            if int(token) in cfg.eos and not ignore_eos:
                finished[b], reasons[b] = True, "stop"
            elif len(generated[b]) >= max_tokens:
                finished[b] = True

        for b in range(n):
            if max_tokens > 0:
                push(b, first[b])
            else:
                finished[b] = True
        emitted = [0] * n

        def emit():
            for b in range(n):
                if on_tokens is not None and len(generated[b]) > emitted[b]:
                    on_tokens(b, generated[b][emitted[b] :])
                emitted[b] = len(generated[b])

        emit()
        cur, pos_dev = self.replicate(first), self.replicate(pos)
        steps = 0
        started = time.perf_counter()
        while not all(finished) and int(pos.max()) + self.args.steps_per_call <= self.context:
            cur, self.caches, out, self.key = self.chunk(
                self.weights,
                self.caches,
                cur,
                pos_dev,
                self.key,
                jnp.float32(temperature),
                jnp.float32(top_p),
            )
            out = np.asarray(out)  # [steps, B]
            pos += self.args.steps_per_call
            pos_dev = pos_dev + self.args.steps_per_call
            steps += self.args.steps_per_call
            for b in range(n):
                for token in out[:, b]:
                    if finished[b]:
                        break
                    push(b, token)
            emit()
        decode_seconds = time.perf_counter() - started
        return generated, reasons, prefill_seconds, decode_seconds, steps


# --- benchmark -----------------------------------------------------------------------------------
def run_bench(mesh, cfg, weights, args, prompt_ids, log):
    """Decode ms/step and aggregate tok/s for B in {1, 2, 4, 8}, plus prefill timings."""
    results = []
    for batch in BENCH_BATCHES:
        if batch > args.batch:
            break
        log(f"bench: batch {batch}, compiling")
        session = Session(mesh, cfg, weights, args, batch, log)
        prefill_notes = []
        for width in sorted({64, 256, 1024, len(prompt_ids)}):
            ids = (list(prompt_ids) * (width // len(prompt_ids) + 1))[:width]
            if width + args.steps_per_call > args.context:
                continue
            session.prefill(ids, 0)  # compile + first run
            _, seconds = session.prefill(ids, 0)
            prefill_notes.append(f"T={width}: {seconds * 1e3:.0f} ms")
        log(f"bench: batch {batch} prefill {', '.join(prefill_notes)}")
        # fill every row with the prompt, warm one device call, then time the rest
        for b in range(batch):
            session.prefill(prompt_ids, b)
        pos = session.replicate(np.full((batch,), len(prompt_ids)))
        cur = session.replicate(np.full((batch,), int(prompt_ids[-1])))
        key = session.replicate_key(jax.random.key(0))
        t, p = jnp.float32(args.temperature), jnp.float32(args.top_p)
        cur, session.caches, out, key = session.chunk(weights, session.caches, cur, pos, key, t, p)
        out.block_until_ready()
        calls = max(1, args.bench_steps // args.steps_per_call)
        started = time.perf_counter()
        for i in range(calls):
            cur, session.caches, out, key = session.chunk(
                weights, session.caches, cur, pos + (i + 1) * args.steps_per_call, key, t, p
            )
        out.block_until_ready()
        seconds = time.perf_counter() - started
        steps = calls * args.steps_per_call
        ms = 1e3 * seconds / steps
        results.append((batch, ms, batch * 1e3 / ms))
        log(
            f"bench: batch {batch}: {steps} steps in {seconds:.2f} s = {ms:.2f} ms/step, "
            f"{batch * 1e3 / ms:.1f} tok/s aggregate"
        )
        del session
    print("\nbatch  ms/step  tok/s (aggregate)")
    for batch, ms, tps in results:
        print(f"{batch:5d}  {ms:7.2f}  {tps:8.1f}")
    return results


# --- main ----------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="pre-sharded int4 container (musespark.load.convert_presharded)",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="HF snapshot directory: tokenizer.json and config.json",
    )
    parser.add_argument(
        "--context", type=int, default=8192, help="KV-cache length (multiple of 128)"
    )
    parser.add_argument("--batch", type=int, default=1, help="decode rows (1, 2, 4 or 8)")
    parser.add_argument("--max-tokens", type=int, default=512, help="generation budget per prompt")
    parser.add_argument("--greedy", action="store_true", help="argmax decoding")
    parser.add_argument("--temperature", type=float, default=sampling.DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=sampling.DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=sampling.DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--prompt", action="append", help="prompt(s) for a non-interactive run")
    parser.add_argument("--raw", action="store_true", help="no chat template (raw completion)")
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=sorted(chat.REASONING_STRENGTH),
        help="chat template setting",
    )
    parser.add_argument("--no-stop", action="store_true", help="ignore the end-of-turn tokens")
    parser.add_argument(
        "--steps-per-call",
        type=int,
        default=64,
        metavar="N",
        help="decode steps per device call (tokens stream once per call)",
    )
    parser.add_argument(
        "--serve",
        default=None,
        metavar="HOST:PORT",
        help="OpenAI-compatible server (one request at a time, row 0)",
    )
    parser.add_argument("--model-name", default="muse-spark-1.2-816b-a42b")
    parser.add_argument(
        "--bench",
        action="store_true",
        help="decode benchmark for B in {1,2,4,8} (up to --batch), then exit",
    )
    parser.add_argument("--bench-steps", type=int, default=256, help="timed decode steps per B")
    args = parser.parse_args()
    if args.context % 128 or args.context < 2 * CHUNK + args.steps_per_call:
        raise ValueError("--context must be a multiple of 128 and hold a prompt plus one call")
    if args.batch not in BENCH_BATCHES:
        raise ValueError("--batch must be 1, 2, 4 or 8")
    if args.bench:
        args.batch = 8
    if args.temperature <= 0:
        args.greedy = True

    started = time.perf_counter()

    def log(message):
        print(f"[{time.perf_counter() - started:6.1f}s] {message}", flush=True)

    devices = jax.devices()
    if len(devices) < TP:
        raise RuntimeError(
            f"need {TP} TPU devices, found {len(devices)} (see scripts/demo_musespark.sh)"
        )
    mesh = jax.sharding.Mesh(np.asarray(devices[:TP]), ("tp",))
    doc = ms_load.read_layout(args.weights)
    cfg = ms_load.config_from_layout(doc)
    if doc["tp"] != TP:
        raise ValueError(f"{args.weights} was converted for tp={doc['tp']}")
    tokenizer = ms_load.load_tokenizer(args.checkpoint)
    log(f"loading weights from {args.weights} ({doc['total_bytes'] / 1e9:.0f} GB)")
    weights = ms_load.load_presharded(mesh, args.weights, cfg, log=log)
    log("weights resident")

    today = datetime.datetime.now().astimezone().date().isoformat()

    def encode(prompt):
        if args.raw:
            return [cfg.bos] + tokenizer.encode(prompt)
        return tokenizer.encode(
            chat.render_chat(prompt, date=today, reasoning_effort=args.reasoning_effort)
        )

    def render_messages(_tok, messages, _mode):
        text = chat.render_messages(messages, date=today, reasoning_effort=args.reasoning_effort)
        return np.asarray(tokenizer.encode(text), np.int32)

    if args.bench:
        run_bench(mesh, cfg, weights, args, encode(args.prompt[0] if args.prompt else "Hello"), log)
        log("bye")
        return

    session = Session(mesh, cfg, weights, args, args.batch, log)
    log("first launches (compiling)")
    generated, *_ = session.generate([encode("Hello")], 4)
    log(f"ready ({tokenizer.decode(generated[0])!r})")

    server = None
    if args.serve:
        host, _, port = args.serve.rpartition(":")
        server = openai_server.OpenAIServer(
            host or "0.0.0.0",
            int(port),
            tokenizer,
            model_name=args.model_name,
            chat_mode="response",
            default_max_tokens=args.max_tokens,
            max_prompt_tokens=args.context - 2 * CHUNK - args.steps_per_call,
            bos_token_id=cfg.bos,
            max_model_len=args.context,
            default_temperature=0.0 if args.greedy else args.temperature,
            default_top_p=args.top_p,
            log=log,
            chat_render=render_messages,
            streamer_factory=lambda tok, **kw: MuseStreamer(tok),
            channel_splitter=split_channels,
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

    while True:
        job = None
        if server is not None:
            job = server.next_job(timeout=1.0)
            if job is None:
                continue
            ids = [int(t) for t in job.tokens]
            budget = args.context - len(ids) - 2 * CHUNK - args.steps_per_call
            if budget <= 0:
                job.fail(f"prompt of {len(ids)} tokens does not fit --context {args.context}")
                continue
            max_tokens, ignore_eos = min(job.max_tokens, budget), job.ignore_eos or args.no_stop
            temperature = job.temperature if not args.greedy else 0.0
            sampling_kwargs = {"temperature": temperature, "top_p": job.top_p, "seed": job.seed}
            on_tokens = lambda row, tokens, job=job: job.emit(tokens)
        else:
            prompt = read_prompt()
            if prompt is None:
                break
            ids = encode(prompt)
            if len(ids) + args.max_tokens + args.steps_per_call > args.context:
                print(
                    f"prompt too long for --context {args.context} "
                    f"with --max-tokens {args.max_tokens}",
                    flush=True,
                )
                ids = ids[: args.context - args.max_tokens - args.steps_per_call]
            max_tokens, ignore_eos, sampling_kwargs = args.max_tokens, args.no_stop, {}
            on_tokens = terminal_printer(MuseStreamer(tokenizer))

        try:
            outs, reasons, prefill_seconds, decode_seconds, steps = session.generate(
                [ids], max_tokens, ignore_eos, on_tokens=on_tokens, **sampling_kwargs
            )
        except Exception as error:
            if job is not None:
                job.fail(f"{type(error).__name__}: {error}")
            raise
        generated, finish_reason = outs[0], reasons[0]
        if job is not None:
            job.complete(
                openai_server.Result(
                    generated=generated,
                    finish_reason=finish_reason,
                    prefill_seconds=prefill_seconds,
                    decode_seconds=decode_seconds,
                    steps=steps,
                )
            )
            log(
                f"{job.kind}: {len(ids)} prompt tokens (prefill {prefill_seconds * 1e3:.0f} ms), "
                f"{len(generated)} generated in {decode_seconds:.2f} s ({steps} steps), "
                f"finish {finish_reason}"
            )
        else:
            n = max(len(generated), 1)
            print(
                f"\n--- {len(ids)} prompt tokens, prefill {prefill_seconds * 1e3:.0f} ms; "
                f"{len(generated)} tokens in "
                f"{decode_seconds:.2f} s: {len(generated) / max(decode_seconds, 1e-9):.1f} tok/s, "
                f"{1e3 * decode_seconds / n:.1f} ms/token "
                f"({steps} device steps, finish {finish_reason})\n",
                flush=True,
            )
    if server is not None:
        server.shutdown()
    log("bye")


if __name__ == "__main__":
    main()
