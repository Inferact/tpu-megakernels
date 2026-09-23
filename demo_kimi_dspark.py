"""Interactive Kimi K3 + DSpark speculative decoding demo on the TP32 megakernel.

Loads the K3 checkpoint (selected layout, tuned recipe) and the DSpark
speculator (fused Pallas draft kernel), then reads prompts from the terminal
on process 0 and streams the generated text while the speculative loop runs.
After each response it prints latency and acceptance metrics, and with
``--baseline`` also decodes the same prompt greedily one token at a time
through the single-row kernel for a speed comparison.

Run under the four-host launcher (``scripts/demo_kimi_dspark.sh``); pass
``--prompt`` one or more times for a non-interactive session.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils as mh
from jax.sharding import NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from kimi import decode_megakernel as model
from kimi import dspark
from kimi import load as kimi_load
import openai_server
from model_paths import resolve_model_source

# Keep the semantic names used below while sourcing both implementations from
# the single public DSpark module.
draft = fused = dspark

DEFAULT_CHECKPOINT = "moonshotai/Kimi-K3"
DEFAULT_DRAFT = "RedHatAI/Kimi-K3-speculator.dspark"
PROMPT_CAPACITY = 8192  # tokens; prompts are broadcast to the other hosts in a fixed-size buffer
JOB_HEADER = 12  # int32 slots before the tokens: flag, token count, max_tokens, score flag, stop-bytes length,
#                  ignore_eos, sampling flag, temperature bits, top_p bits, seed
STOP_CAPACITY = 2048  # bytes of JSON-encoded stop strings after the tokens
CHAT_MODES = ("none", "response", "think")
TERMINAL_STREAM_CHUNK_TOKENS = 4


def _resolve_draft(path):
    path = Path(path)
    if (path / "model.safetensors").exists():
        return path
    snapshots = sorted(p for p in path.glob("*") if (p / "model.safetensors").exists())
    if not snapshots:
        raise FileNotFoundError(f"No speculator snapshot with model.safetensors under {path}")
    return snapshots[-1]


def render_chat_tokens(tokenizer, prompt, mode):
    """Render one user turn and the assistant generation prefix in K3's XTML format."""
    control = lambda text: tokenizer.encode(text, allowed_special="all")
    text = lambda text: tokenizer.encode(text, disallowed_special=())
    open_tag = lambda tag, attrs="": control("<|open|>") + text(tag + attrs) + control("<|sep|>")
    close_tag = lambda tag: control("<|close|>") + text(tag) + control("<|sep|>")
    end = control("<|end_of_msg|>")

    tokens = []
    if mode == "think":
        body = (
            "`thinking_effort` guides on how much to think in your thinking channel (not including the "
            "response channel), supported values include `low`, `medium`, `high`, and `max`.\n"
            "Now the system is invoked with `thinking_effort=max`."
        )
        tokens += (
            open_tag("message", ' role="system" type="thinking-effort"')
            + text(body)
            + close_tag("message")
            + end
        )
    tokens += open_tag("message", ' role="user"') + text(prompt) + close_tag("message") + end
    tokens += open_tag("message", ' role="assistant"') + open_tag(
        "think" if mode == "think" else "response"
    )
    return np.asarray(tokens, np.int32)


KERNEL_DEFAULTS = frozenset(
    {
        "mla_cache_write_in_attend",
        "mla_pv_mxu_hilo",
    }
)


def decode_options(batch_size: int, recipe: str = "plain") -> dict:
    """Kernel options used by the public demo's plain and tuned decode paths."""
    options = {
        "gate_first_kda_projection": True,
        "kernel_options": tuple(sorted(KERNEL_DEFAULTS)),
    }
    if recipe == "plain":
        return options
    if recipe != "tuned":
        raise ValueError("recipe must be 'plain' or 'tuned'")
    eight = batch_size == 8
    return {
        **options,
        "residual_reduction_order": "relaxed",
        "mla_cache_batch_tile_size": math.gcd(4, batch_size),
        "alias_expert_scales": batch_size >= 6,
        "attention_token_scatter": eight,
        "moe_token_scatter_output": eight,
        "moe_routed_reduce_scatter": batch_size in (6, 8),
        "sequence_parallel_pre_attention": eight,
    }


def _log(message):
    if jax.process_index() == 0:
        print(message, flush=True)


def _host(value):
    return np.asarray(value.addressable_shards[0].data)


def _prompt_stream():
    """The launcher's stdin: ``distributed_entry.py`` duplicates fd 0 before JAX
    starts (``K3_STDIN_FD``), so a later redirection of fd 0 cannot cut it off."""
    import os

    fd = os.environ.get("K3_STDIN_FD")
    if fd is None:
        return sys.stdin
    try:
        return os.fdopen(int(fd), "r", closefd=False)
    except OSError:
        return sys.stdin


class StartupDisplay:
    """One live status line on process 0 while the demo starts up.

    Weight loading reports expert layers as they land; the line shows the
    fraction done, the elapsed time and an ETA. Other stages show elapsed
    time only. On a terminal the line is rewritten in place; in a log file a
    fresh line is printed every 15 seconds.
    """

    def __init__(self, enabled, expert_layers, ranks, mode="auto"):
        import threading

        self.enabled = enabled
        self.expert_layers = expert_layers
        self.ranks = ranks
        self.lock = threading.Lock()
        self.layers_done = {}
        self.ranks_done = 0
        self.families_done = 0
        self.stage_name = "starting"
        self.stage_started = time.perf_counter()
        self.started = self.stage_started
        self.background = []
        self.global_layers = None  # (total layers done across hosts, per-host layers done)
        self.tty = enabled and (mode == "live" or (mode == "auto" and sys.stdout.isatty()))
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        if enabled:
            self.thread.start()

    def stage(self, name):
        with self.lock:
            self.stage_name = name
            self.stage_started = time.perf_counter()
        self._render(force=True)

    def note(self, text):
        with self.lock:
            self.background.append(text)

    def update_global(self, total_layers, per_host):
        with self.lock:
            self.global_layers = (total_layers, per_host)

    def event(self, event):
        with self.lock:
            if event["kind"] == "layer":
                self.layers_done[event["rank"]] = event["layer"]
            elif event["kind"] == "rank":
                self.ranks_done += 1
            elif event["kind"] == "family":
                self.families_done += 1

    def _line(self):
        elapsed = time.perf_counter() - self.started
        with self.lock:
            stage = self.stage_name
            done = sum(self.layers_done.values())
            total = self.expert_layers * self.ranks
            notes = "; ".join(self.background[-4:])
            global_layers = self.global_layers
        text = f"[{elapsed:5.0f} s] {stage}"
        if stage.startswith("loading target"):
            if global_layers is not None:
                hosts = len(global_layers[1])
                done, total = global_layers[0], total * hosts
                slowest = min(global_layers[1]) / (self.expert_layers * self.ranks)
            fraction = done / total if total else 0.0
            # The step finishes when the slowest host does.
            pace_fraction = slowest if global_layers is not None else fraction
            eta = (time.perf_counter() - self.stage_started) * (1 - pace_fraction) / pace_fraction if pace_fraction > 0.02 else None
            text += f": expert layers {done}/{total} ({100 * fraction:3.0f}%)"
            if global_layers is not None:
                text += f" over {hosts} hosts, slowest host {100 * slowest:3.0f}%"
            else:
                text += f", {self.ranks_done}/{self.ranks} ranks resident"
            text += f", about {eta:.0f} s left" if eta is not None else ""
        elif stage.startswith("waiting for the other hosts") and global_layers is not None:
            slowest = min(global_layers[1]) / (self.expert_layers * self.ranks)
            text += f": slowest host at {100 * slowest:3.0f}% ({time.perf_counter() - self.stage_started:.0f} s)"
        else:
            text += f" ({time.perf_counter() - self.stage_started:.0f} s)"
        if notes:
            text += f"  |  {notes}"
        return text

    def _render(self, force=False):
        if not self.enabled:
            return
        text = self._line()
        if self.tty:
            sys.stdout.write("\r" + text.ljust(140)[:140])
        else:
            sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def _run(self):
        interval = 1.0 if self.tty else 15.0
        while not self.stop.wait(interval):
            self._render()

    def finish(self, text):
        self.stop.set()
        if self.enabled:
            if self.thread.is_alive():
                self.thread.join()
            with self.lock:
                notes = list(self.background)
            sys.stdout.write(("\r" if self.tty else "") + text.ljust(140) + "\n")
            if notes:
                sys.stdout.write("    stages: " + "; ".join(notes) + "\n")
            sys.stdout.flush()


def _abstract(shape, dtype, sharding):
    return jax.ShapeDtypeStruct(tuple(shape), dtype, sharding=sharding)


def _layout_mismatch(tree, abstract):
    """None when every leaf of ``tree`` has the shape, dtype and sharding the programs
    were compiled for, else a description of the first difference."""
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    expected = jax.tree.leaves(abstract)
    if len(leaves) != len(expected):
        return f"{len(leaves)} arrays vs {len(expected)} compiled"
    for (path, leaf), spec in zip(leaves, expected):
        name = jax.tree_util.keystr(path)
        if tuple(leaf.shape) != tuple(spec.shape) or leaf.dtype != spec.dtype:
            return f"{name}: {leaf.shape} {leaf.dtype} vs compiled {spec.shape} {spec.dtype}"
        if spec.sharding is not None and not leaf.sharding.is_equivalent_to(spec.sharding, leaf.ndim):
            return f"{name}: sharding {leaf.sharding} vs compiled {spec.sharding}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HF snapshot (tokenizer, and weights unless --weights)")
    parser.add_argument("--draft-weights", default=None,
                        help="Speculator in the pre-sharded kernel layout (default: "
                             "$KIMI_DRAFT_PRESHARDED_WEIGHTS when set, else built from --draft)")
    parser.add_argument("--weights", default=None,
                        help="Weights source: a pre-sharded kernel-layout directory or the HF snapshot "
                             "(default: $KIMI_PRESHARDED_WEIGHTS when set, else --checkpoint)")
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--prompt", action="append", help="Prompt(s) for a non-interactive session; repeatable")
    parser.add_argument("--chat", choices=CHAT_MODES[:3], default="response",
                        help="K3 chat format: 'response' (no thinking), 'think', or 'none' (raw completion)")
    parser.add_argument("--context", type=int, default=2048, help="Attention cache length (multiple of 128)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Generation budget per prompt")
    parser.add_argument("--steps-per-call", type=int, default=4,
                        help="Decode steps per device call")
    parser.add_argument(
        "--fixed-acceptance-length", type=int, choices=range(1, 9), default=None, metavar="[1-8]",
        help="Force tokens emitted per speculative step (benchmarking only; default: use actual acceptance)",
    )
    parser.add_argument("--recipe", choices=("plain", "tuned"), default="tuned")
    parser.add_argument("--ranks-in-flight", type=int, default=8, help="Local ranks loaded concurrently")
    parser.add_argument("--baseline", action="store_true", help="Also decode greedily one token at a time (B1)")
    parser.add_argument("--target-only", action="store_true",
                        help="Serve with the Kimi megakernel instead of DSpark speculative decoding")
    parser.add_argument(
        "--target-batch-size", type=int, choices=range(1, 9), default=1, metavar="[1-8]",
        help="Number of concurrent requests decoded by each target-only megakernel invocation",
    )
    parser.add_argument("--no-stop", action="store_true", help="Ignore end-of-message tokens")
    parser.add_argument("--inbox", type=Path, default=None,
                        help="Read prompts appended to this file (one per line) instead of stdin, e.g. when "
                             "srun does not forward the terminal; write 'quit' to end")
    parser.add_argument("--progress", choices=("auto", "live", "log"), default="auto",
                        help="Start-up progress: rewrite one line ('live', for a terminal) or print a line every 15 s ('log')")
    parser.add_argument("--serve", default=None, metavar="HOST:PORT",
                        help="Server mode: answer OpenAI-compatible /v1/completions and /v1/chat/completions requests "
                             "(one request or target-only batch at a time; greedy unless the request sets temperature > 0) instead of "
                             "reading prompts; --max-tokens is the default budget")
    parser.add_argument("--model-name", default="kimi-k3", help="Model id reported by the server")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy, the fast path). > 0 samples from the top-p nucleus "
                             "with exact speculative rejection sampling; in server mode this is the default a request "
                             "without 'temperature' gets")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus mass when sampling (server default too)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Sampling seed for the prompts of this run (default: a fresh random seed per prompt)")
    args = parser.parse_args()
    if args.steps_per_call < 1:
        raise ValueError("--steps-per-call must be positive")
    if args.target_only and args.fixed_acceptance_length is not None:
        raise ValueError("--fixed-acceptance-length only applies to DSpark decoding")
    if not args.target_only and args.target_batch_size != 1:
        raise ValueError("--target-batch-size only applies with --target-only")
    if args.target_batch_size != 1 and not args.serve:
        raise ValueError("--target-batch-size greater than one requires --serve")
    args.baseline = args.baseline or args.target_only
    rows = 8  # anchor + 7 drafts, greedy acceptance
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("--temperature must be >= 0 and --top-p in (0, 1]")
    sampling_enabled = bool(args.serve) or args.temperature > 0  # compile the sampling programs
    checkpoint_revision = (
        os.environ.get("KIMI_CHECKPOINT_REVISION")
        if args.checkpoint == DEFAULT_CHECKPOINT
        else None
    )
    draft_revision = (
        os.environ.get("KIMI_DRAFT_REVISION") if args.draft == DEFAULT_DRAFT else None
    )
    args.checkpoint = str(resolve_model_source(args.checkpoint, revision=checkpoint_revision))
    args.draft = str(resolve_model_source(args.draft, revision=draft_revision))
    presharded = os.environ.get("KIMI_PRESHARDED_WEIGHTS")
    draft_presharded = os.environ.get("KIMI_DRAFT_PRESHARDED_WEIGHTS")
    if args.weights is None:
        args.weights = (
            presharded
            if presharded and kimi_load.is_presharded(presharded)
            else args.checkpoint
        )
    if (
        args.draft_weights is None
        and draft_presharded
        and (Path(draft_presharded) / "index.json").is_file()
    ):
        args.draft_weights = draft_presharded
    if args.context < 384 or args.context % 128:
        raise ValueError("--context must be at least 384 and a multiple of 128")

    import queue
    import threading

    mesh, _, _ = model._logical_mesh("0,1,2,3,4")
    replicated = NamedSharding(mesh, P())
    sharded = NamedSharding(mesh, P("tp"))
    layers = 93
    if args.target_only:
        draft_dir = None
        c = draft.DraftConfig()
        aux_layers = ()
    else:
        draft_dir = _resolve_draft(args.draft)
        c = draft.DraftConfig.from_checkpoint(draft_dir)
        aux_layers = c.aux_layers
    drafts = rows - 1
    local_ranks = sum(1 for device in mesh.devices.flat if device.process_index == jax.process_index())
    display = StartupDisplay(jax.process_index() == 0, layers - 1, local_ranks, args.progress)

    def put(value, dtype=np.int32):
        return jax.device_put(np.asarray(value, dtype), replicated)

    # --- programs (traced now, compiled in the background against abstract inputs) ---
    # The verify takes the accepted row of the previous step as its fifth
    # argument and reads that row's KDA states itself (no state copy).
    verify = model.make_decode(
        mesh, batch_size=rows, layers=layers, donate=False, sequence_rows=rows,
        aux_hidden_layers=aux_layers, select_state_row=True, **decode_options(rows, args.recipe),
    )
    fused_step = fused.make_fused_draft(c, mesh, rows)
    sample_rows = draft.make_row_sampler(mesh, rows, c.vocab)
    row_offsets = jnp.arange(rows, dtype=jnp.int32)

    def step_uniforms(seed, position):
        """The two uniforms per row of one verify step: a function of the seed and the step's first
        row position (positions never repeat within a generation), identical on every rank and host."""
        key = jax.random.fold_in(jax.random.PRNGKey(seed), position)
        return jax.random.uniform(key, (rows, 2), jnp.float32)
    kda_layers = layers - layers // 4 - 1
    mla_layers = layers // 4 + 1

    def state_shapes(batch, sequence_rows=None):
        sequence_rows = batch if sequence_rows is None else sequence_rows
        sequences = batch // sequence_rows
        return (
            ((32, batch, kda_layers, 3, 3, 3, 128), jnp.bfloat16),
            ((32, batch, kda_layers, 3, 128, 128), jnp.float32),
            ((32, sequences, mla_layers, 3, args.context, 256), jnp.bfloat16),
            ((32, sequences, mla_layers, 3, args.context, 128), jnp.bfloat16),
        )

    def zero_states(batch, sequence_rows=None):
        return tuple(
            jax.jit(lambda shape=shape, dtype=dtype: jnp.zeros(shape, dtype), out_shardings=sharded)()
            for shape, dtype in state_shapes(batch, sequence_rows)
        )

    def prefill_chunk(model_weights, kernel_weights, target_states, kcache, kpositions, vector, row_tokens, position,
                      state_row, count):
        """One 8-token prompt chunk through the verify kernel and the fused draft (context update).
        ``state_row`` is the accepted row of the previous chunk; the caller passes ``count`` on as the next."""
        logits, aux, new_states = verify(model_weights, target_states, row_tokens, position + row_offsets, state_row)
        vector, kcache, kpositions, _ = fused_step(kernel_weights, kcache, kpositions, aux, logits, vector, position, count)
        return new_states, kcache, kpositions, vector

    def score_rows_local(logits, targets):
        """Per row: log-probability of ``targets``, the greedy id and its log-probability,
        from vocabulary-sharded ``[32, rows, vocab / 32]`` logits (one small all-gather)."""
        local = logits[0].astype(jnp.float32)  # [rows, vocab / 32]
        rank = jax.lax.axis_index("tp")
        width = local.shape[1]
        local_max = jnp.max(local, axis=1)
        local_lse = jnp.log(jnp.sum(jnp.exp(local - local_max[:, None]), axis=1))  # relative to local_max
        local_id = jnp.argmax(local, axis=1).astype(jnp.int32)
        offset = targets - rank * width
        owned = (offset >= 0) & (offset < width)
        target_logit = jnp.where(
            owned, jnp.take_along_axis(local, jnp.clip(offset, 0, width - 1)[:, None], axis=1)[:, 0], -jnp.inf
        )
        maxima, lses, ids, targets_logit = jax.lax.all_gather(
            (local_max, local_lse, local_id + rank * width, target_logit), "tp", axis=0
        )  # [32, rows]
        best = jnp.max(maxima, axis=0)
        lse = best + jnp.log(jnp.sum(jnp.exp(maxima - best[None] + lses), axis=0))
        greedy = jnp.take_along_axis(ids, jnp.argmax(maxima, axis=0)[None], axis=0)[0]
        return jnp.max(targets_logit, axis=0) - lse, greedy, best - lse

    score_rows = jax.shard_map(score_rows_local, mesh=mesh, in_specs=(P("tp"), P()), out_specs=(P(), P(), P()), check_vma=False)

    def prefill_chunk_scored(model_weights, kernel_weights, target_states, kcache, kpositions, vector, row_tokens, position,
                             state_row, count, targets):
        """``prefill_chunk`` that also scores each row's distribution at ``targets`` (the next prompt tokens)."""
        logits, aux, new_states = verify(model_weights, target_states, row_tokens, position + row_offsets, state_row)
        vector, kcache, kpositions, _ = fused_step(kernel_weights, kcache, kpositions, aux, logits, vector, position, count)
        target_logprob, greedy, greedy_logprob = score_rows(logits, targets)
        return new_states, kcache, kpositions, vector, target_logprob, greedy, greedy_logprob

    def prefill_chunk_sampled(model_weights, kernel_weights, target_states, kcache, kpositions, vector, row_tokens, position,
                              state_row, count, targets, temperature, top_p, seed):
        """``prefill_chunk_scored`` for the last prompt chunk when sampling: the bonus token (the first
        generated token) is drawn from the top-p nucleus of row ``count`` instead of being its argmax."""
        logits, aux, new_states = verify(model_weights, target_states, row_tokens, position + row_offsets, state_row)
        no_drafts = jnp.full((rows,), -1, jnp.int32)
        _, bonus = sample_rows(logits, no_drafts, count, temperature, top_p, step_uniforms(seed, position))
        vector, kcache, kpositions, _ = fused_step(kernel_weights, kcache, kpositions, aux, logits, vector, position, count, bonus)
        target_logprob, greedy, greedy_logprob = score_rows(logits, targets)
        return new_states, kcache, kpositions, vector, target_logprob, greedy, greedy_logprob

    def _speculate(model_weights, kernel_weights, target_states, state_row, kcache, kpositions, vector, position, sampling):
        """``steps_per_call`` speculative steps; returns emitted tokens [steps, rows] (-1 padded).
        ``sampling`` is None (greedy acceptance inside the fused kernel) or ``(temperature, top_p, seed)``:
        then the acceptance and the bonus token come from ``sample_rows`` (exact top-p rejection sampling
        against the greedy drafts) and are forced on the fused kernel."""
        forced_count = jnp.int32(
            -1 if args.fixed_acceptance_length is None
            else args.fixed_acceptance_length - 1
        )

        def step(index, carry):
            (model_weights, kernel_weights, target_states, state_row, kcache, kpositions, vector, position,
             emitted, accepted) = carry
            anchor = vector[0, fused.OUT_BONUS]
            proposals = vector[0, fused.OUT_DRAFTS:fused.OUT_DRAFTS + rows]
            row_tokens = jnp.concatenate((anchor[None], proposals[:drafts])).astype(jnp.int32)
            logits, aux, new_states = verify(model_weights, target_states, row_tokens, position + row_offsets, state_row)
            if sampling is None:
                forced = (forced_count,)
            else:
                temperature, top_p, seed = sampling
                row_drafts = jnp.concatenate((row_tokens[1:], jnp.full((1,), -1, jnp.int32)))  # row k verifies draft k
                forced = sample_rows(logits, row_drafts, jnp.int32(-1), temperature, top_p, step_uniforms(seed, position))
            vector, kcache, kpositions, _ = fused_step(kernel_weights, kcache, kpositions, aux, logits, vector, position, *forced)
            count = vector[0, fused.OUT_COUNT]
            bonus = vector[0, fused.OUT_BONUS]
            slot = jnp.arange(rows)
            step_tokens = jnp.where(
                slot < count, row_tokens[jnp.minimum(slot + 1, rows - 1)], jnp.where(slot == count, bonus, -1)
            )
            return (model_weights, kernel_weights, new_states, count, kcache, kpositions, vector, position + count + 1,
                    emitted.at[index].set(step_tokens), accepted.at[index].set(count))

        carry = (model_weights, kernel_weights, target_states, state_row, kcache, kpositions, vector, position,
                 jnp.full((args.steps_per_call, rows), -1, jnp.int32), jnp.zeros((args.steps_per_call,), jnp.int32))
        result = jax.lax.fori_loop(0, args.steps_per_call, step, carry)
        return result[2:]

    def speculate(model_weights, kernel_weights, target_states, state_row, kcache, kpositions, vector, position):
        return _speculate(model_weights, kernel_weights, target_states, state_row, kcache, kpositions, vector, position, None)

    def speculate_sampled(model_weights, kernel_weights, target_states, state_row, kcache, kpositions, vector, position,
                          temperature, top_p, seed):
        return _speculate(model_weights, kernel_weights, target_states, state_row, kcache, kpositions, vector, position,
                          (temperature, top_p, seed))

    if args.target_only:
        target_batch_size = args.target_batch_size
        target = model.make_decode(
            mesh, batch_size=target_batch_size, layers=layers, donate=False,
            sequence_rows=1, **decode_options(target_batch_size, args.recipe),
        )
        target_argmax = draft.make_sharded_argmax(mesh, target_batch_size, c.vocab)

        def greedy_token(model_weights, target_states, token, position):
            logits, target_states = target(model_weights, target_states, token, position)
            return target_states, target_argmax(logits).astype(jnp.int32)

        def greedy_decode(model_weights, target_states, token, position):
            def step(index, carry):
                target_states, token, out = carry
                logits, target_states = target(model_weights, target_states, token, position + index)
                token = target_argmax(logits).astype(jnp.int32)
                return target_states, token, out.at[index].set(token)

            target_states, token, output = jax.lax.fori_loop(
                0, args.steps_per_call, step,
                (target_states, token, jnp.zeros((args.steps_per_call, target_batch_size), jnp.int32)),
            )
            return target_states, token, output, position + args.steps_per_call
    elif args.baseline:
        single = model.make_decode(mesh, batch_size=1, layers=layers, donate=False, **decode_options(1, "plain"))
        single_argmax = draft.make_sharded_argmax(mesh, 1, c.vocab)

        def greedy_token(model_weights, target_states, token, position):
            logits, target_states = single(model_weights, target_states, token[None], position[None])
            return target_states, single_argmax(logits).astype(jnp.int32)[0]

        def greedy_decode(model_weights, target_states, token, position):
            def step(index, carry):
                target_states, token, out = carry
                logits, target_states = single(model_weights, target_states, token[None], (position + index)[None])
                token = single_argmax(logits).astype(jnp.int32)[0]
                return target_states, token, out.at[index].set(token)

            steps = args.steps_per_call if args.target_only else args.steps_per_call * rows
            target_states, token, output = jax.lax.fori_loop(
                0, steps, step, (target_states, token, jnp.zeros((steps,), jnp.int32))
            )
            return target_states, token, output, position + steps

    programs = {}
    compile_error = []

    # A tiny cross-host collective program: the runtime's first collective
    # launch carries a one-time set-up cost that is better paid during loading.
    collective_probe = jax.jit(jax.shard_map(
        lambda x: jax.lax.psum(x, "tp"), mesh=mesh, in_specs=P("tp"), out_specs=P(), check_vma=False,
    ))

    def run_probe(label):
        started = time.perf_counter()
        jax.block_until_ready(collective_probe(jax.device_put(np.ones((32, 8, 128), np.float32), sharded)))
        display.note(f"{label} collective {time.perf_counter() - started:.1f} s")

    # Pacer: the runtime defers work created by the loader (thousands of
    # donated multi-GB buffers) and flushes it on the next execution, a
    # 30 s stall. Executing the tiny collective every few seconds during the
    # load drains that backlog incrementally. Every process runs the same
    # probe sequence; the probe sums a per-process done flag so all hosts stop
    # at the same iteration (a mismatched collective order would deadlock).
    load_done = {"value": 0.0}
    pacer_stats = {"max": 0.0, "count": 0}

    process_count = jax.process_count()

    def pace():
        while True:
            # Lane 0: done flag; lane 1: this host's expert layers loaded;
            # lane 2 + p: the same, one lane per process (for the slowest host).
            with display.lock:
                layers_loaded = float(sum(display.layers_done.values()))
            row = np.zeros((128,), np.float32)
            row[0] = load_done["value"]
            row[1] = layers_loaded
            row[2 + jax.process_index()] = layers_loaded
            flag = jax.make_array_from_callback(
                (32, 8, 128), sharded, lambda index, row=row: np.broadcast_to(row, (1, 8, 128)).copy()
            )
            started = time.perf_counter()
            summed = np.asarray(collective_probe(flag).addressable_shards[0].data).reshape(-1, 128)[0] / local_ranks
            elapsed = time.perf_counter() - started
            pacer_stats["max"] = max(pacer_stats["max"], elapsed)
            pacer_stats["count"] += 1
            display.update_global(int(summed[1]), [int(summed[2 + p]) for p in range(process_count)])
            if summed[0] >= process_count:  # every process has finished loading
                break
            time.sleep(2.0)
        display.note(f"pacer: {pacer_stats['count']} probes, longest {pacer_stats['max']:.1f} s")

    pacer_thread = threading.Thread(target=pace, daemon=True)

    def compile_programs():
        try:
            started = time.perf_counter()
            weights_abs = kimi_load.abstract_weights(mesh, args.weights, layers=layers)
            scalar_abs = _abstract((), jnp.int32, replicated)
            if args.target_only:
                target_abs = tuple(
                    _abstract(shape, dtype, sharded)
                    for shape, dtype in state_shapes(target_batch_size, sequence_rows=1)
                )
                target_rows_abs = _abstract((target_batch_size,), jnp.int32, replicated)
                programs["greedy_token"] = jax.jit(greedy_token).lower(
                    weights_abs, target_abs, target_rows_abs, target_rows_abs
                ).compile()
                programs["greedy_decode"] = jax.jit(greedy_decode).lower(
                    weights_abs, target_abs, target_rows_abs, target_rows_abs
                ).compile()
                programs["abstract"] = weights_abs
                display.note(f"target programs compiled in {time.perf_counter() - started:.0f} s")
                started = time.perf_counter()
                programs["warm_single"] = zero_states(target_batch_size, sequence_rows=1)
                jax.block_until_ready(programs["warm_single"])
                display.note(f"state buffers in {time.perf_counter() - started:.0f} s")
                run_probe("early")
                pacer_thread.start()
                return

            kernel_abs = {
                name: _abstract((32, *shape), dtype, sharded)
                for name, (shape, dtype) in fused.kernel_weight_shapes(c, max_position=args.context + 64).items()
            }
            states_abs = tuple(_abstract(shape, dtype, sharded) for shape, dtype in state_shapes(rows))
            kcache_abs = _abstract((32, c.layers, c.window, 2 * c.head_dim), jnp.bfloat16, sharded)
            kpositions_abs = _abstract((1, c.window), jnp.int32, replicated)
            vector_abs = _abstract((1, 128), jnp.int32, replicated)
            scalar_abs = _abstract((), jnp.int32, replicated)
            float_abs = _abstract((), jnp.float32, replicated)
            rows_abs = _abstract((rows,), jnp.int32, replicated)
            sampled_error = []

            def compile_sampled():
                # Compile these in the background because they substantially delay readiness and
                # greedy requests do not need them. The first sampled request waits for this thread
                # through ensure_sampled_programs.
                try:
                    programs["prefill_chunk_sampled"] = jax.jit(prefill_chunk_sampled).lower(
                        weights_abs, kernel_abs, states_abs, kcache_abs, kpositions_abs, vector_abs, rows_abs, scalar_abs,
                        scalar_abs, scalar_abs, rows_abs, float_abs, float_abs, scalar_abs,
                    ).compile()
                    programs["speculate_sampled"] = jax.jit(speculate_sampled).lower(
                        weights_abs, kernel_abs, states_abs, scalar_abs, kcache_abs, kpositions_abs, vector_abs, scalar_abs,
                        float_abs, float_abs, scalar_abs,
                    ).compile()
                except Exception as error:  # noqa: BLE001 - reported by the caller
                    sampled_error.append(error)

            if sampling_enabled:
                programs["sampled_thread"] = threading.Thread(target=compile_sampled, daemon=True)
                programs["sampled_error"] = sampled_error
                programs["sampled_thread"].start()
            programs["prefill_chunk"] = jax.jit(prefill_chunk).lower(
                weights_abs, kernel_abs, states_abs, kcache_abs, kpositions_abs, vector_abs, rows_abs, scalar_abs,
                scalar_abs, scalar_abs,
            ).compile()
            programs["speculate"] = jax.jit(speculate).lower(
                weights_abs, kernel_abs, states_abs, scalar_abs, kcache_abs, kpositions_abs, vector_abs, scalar_abs,
            ).compile()
            if args.serve:
                programs["prefill_chunk_scored"] = jax.jit(prefill_chunk_scored).lower(
                    weights_abs, kernel_abs, states_abs, kcache_abs, kpositions_abs, vector_abs, rows_abs, scalar_abs,
                    scalar_abs, scalar_abs, rows_abs,
                ).compile()
            if args.baseline:
                single_abs = tuple(_abstract(shape, dtype, sharded) for shape, dtype in state_shapes(1))
                programs["greedy_token"] = jax.jit(greedy_token).lower(weights_abs, single_abs, scalar_abs, scalar_abs).compile()
                programs["greedy_decode"] = jax.jit(greedy_decode).lower(weights_abs, single_abs, scalar_abs, scalar_abs).compile()
            programs["abstract"] = (weights_abs, kernel_abs)
            display.note(f"programs compiled in {time.perf_counter() - started:.0f} s")
            # The zero states and the empty draft cache do not depend on the
            # weights: build (and so compile) them here for the first launches.
            started = time.perf_counter()
            programs["warm_states"] = zero_states(rows)
            programs["warm_cache"] = fused.empty_kernel_cache(c, mesh)
            if args.baseline:
                programs["warm_single"] = zero_states(1)
            jax.block_until_ready((programs["warm_states"], programs["warm_cache"]))
            display.note(f"state buffers in {time.perf_counter() - started:.0f} s")
            run_probe("early")
            pacer_thread.start()
        except Exception as error:  # fall back to compiling against the real arrays later
            compile_error.append(error)
            display.note(f"background compile failed ({type(error).__name__}); will compile after loading")

    draft_result = {}

    def load_draft():
        started = time.perf_counter()
        if args.draft_weights is not None:
            # RoPE rows are cheap to construct but make an otherwise fixed
            # pre-sharded checkpoint context-length-specific. Load the stored
            # tables only as placeholders, then replace them with exactly the
            # number of rows required by this invocation.
            draft_weights = fused.load_kernel_shards(args.draft_weights, mesh)
            rope_q, rope_kv = fused.rope_tables(c, args.context + 96)
            dynamic_rope = fused.place_kernel_weights(
                {
                    "rope_q": np.broadcast_to(rope_q, (mesh.size, *rope_q.shape)),
                    "rope_kv": np.broadcast_to(rope_kv, (mesh.size, *rope_kv.shape)),
                },
                mesh,
            )
            draft_weights.update(dynamic_rope)
            draft_result["weights"] = draft_weights
        else:
            host_draft = draft.load_checkpoint_weights(draft_dir, c)
            draft_result["weights"] = fused.place_kernel_weights(
                fused.kernel_weight_shards(c, host_draft, max_position=args.context + 64), mesh
            )
        jax.block_until_ready(draft_result["weights"])
        display.note(f"draft weights resident in {time.perf_counter() - started:.0f} s")

    compile_thread = threading.Thread(target=compile_programs, daemon=True)
    draft_thread = None if args.target_only else threading.Thread(target=load_draft, daemon=True)
    compile_thread.start()
    if draft_thread is not None:
        draft_thread.start()

    # --- target weights (streamed, FP8 gate/up converted on device per layer) -------
    display.stage("loading target weights" + (" (pre-sharded)" if kimi_load.is_presharded(args.weights) else ""))
    started = time.perf_counter()
    weights = kimi_load.load_weights(
        args.weights, mesh, layers=layers, ranks_in_flight=args.ranks_in_flight, progress=display.event,
    )
    jax.block_until_ready(weights)
    target_seconds = time.perf_counter() - started
    display.stage("waiting for program compilation" if args.target_only else "waiting for the draft weights and program compilation")
    if draft_thread is not None:
        draft_thread.join()
    compile_thread.join()
    load_done["value"] = 1.0
    if pacer_thread.is_alive():
        display.stage("waiting for the other hosts to finish loading")
        pacer_thread.join()
    kernel_weights = None if args.target_only else draft_result["weights"]
    display.stage("loading the tokenizer")
    phase_started = time.perf_counter()
    tokenizer = kimi_load.load_tokenizer(args.checkpoint)
    display.note(f"tokenizer {time.perf_counter() - phase_started:.1f} s")
    end_tokens = {int(tokenizer.encode(t, allowed_special="all")[0]) for t in ("[EOS]", "<|end_of_msg|>")}

    compiled_inputs = weights if args.target_only else (weights, kernel_weights)
    mismatch = _layout_mismatch(compiled_inputs, programs["abstract"]) if "abstract" in programs else "no background compile"
    if mismatch is None:
        prefill_chunk_sampled_program = speculate_sampled_program = None
        if args.target_only:
            greedy_token_program = programs["greedy_token"]
            greedy_decode_program = programs["greedy_decode"]
        else:
            prefill_chunk_program = programs["prefill_chunk"]
            speculate_program = programs["speculate"]
            prefill_chunk_scored_program = programs.get("prefill_chunk_scored")
            if args.baseline:
                greedy_token_program = programs["greedy_token"]
                greedy_decode_program = programs["greedy_decode"]
    else:
        display.note(f"recompiling: {mismatch}")
        display.stage("compiling programs against the loaded weights")
        prefill_chunk_sampled_program = jax.jit(prefill_chunk_sampled) if sampling_enabled else None
        speculate_sampled_program = jax.jit(speculate_sampled) if sampling_enabled else None
        if args.target_only:
            greedy_token_program = jax.jit(greedy_token)
            greedy_decode_program = jax.jit(greedy_decode)
        else:
            prefill_chunk_program = jax.jit(prefill_chunk)
            speculate_program = jax.jit(speculate)
            prefill_chunk_scored_program = jax.jit(prefill_chunk_scored) if args.serve else None
            if args.baseline:
                greedy_token_program = jax.jit(greedy_token)
                greedy_decode_program = jax.jit(greedy_decode)

    def ensure_sampled_programs():
        """First sampled request: wait for the background compile of the sampling programs (every host
        joins its own thread; the wait is host-local and the first launch then happens in lockstep)."""
        nonlocal prefill_chunk_sampled_program, speculate_sampled_program
        thread = programs.pop("sampled_thread", None)
        if thread is not None:
            started = time.perf_counter()
            thread.join()
            if programs["sampled_error"]:
                raise programs["sampled_error"][0]
            if mismatch is None:
                prefill_chunk_sampled_program = programs["prefill_chunk_sampled"]
                speculate_sampled_program = programs["speculate_sampled"]
            _log(f"sampling programs ready (waited {time.perf_counter() - started:.0f} s)")
        if prefill_chunk_sampled_program is None:
            raise RuntimeError("sampling programs were not compiled (start with --serve or --temperature > 0)")

    def sampling_scalars(sampling):
        temperature, top_p, seed = sampling
        return put(temperature, np.float32).reshape(()), put(top_p, np.float32).reshape(()), put(seed).reshape(())

    def prefill(target_states, kcache, kpositions, prompt_tokens, score=False, sampling=None):
        """Run the prompt through the verify kernel in 8-token chunks.

        With ``score`` also returns, for prompt positions 1 .. length - 1, the
        log-probability of the prompt token, the greedy token and its
        log-probability (from the distribution at the previous position).
        With ``sampling`` (``(temperature, top_p, seed)``) the first generated
        token is sampled from the last position's top-p nucleus."""
        length = len(prompt_tokens)
        padded = np.full((-(-length // rows) * rows + 1,), int(prompt_tokens[-1]), np.int32)
        padded[:length] = prompt_tokens
        vector = put(np.full((1, 128), -1, np.int32))
        state_row = put(0).reshape(())
        scores = []
        starts = list(range(0, len(padded) - 1, rows))
        for start in starts:
            count = min(rows, length - start) - 1
            if sampling is not None and start == starts[-1]:
                target_states, kcache, kpositions, vector, target_logprob, greedy, greedy_logprob = prefill_chunk_sampled_program(
                    weights, kernel_weights, target_states, kcache, kpositions, vector,
                    put(padded[start:start + rows]), put(start).reshape(()), state_row, put(count).reshape(()),
                    put(padded[start + 1:start + rows + 1]), *sampling_scalars(sampling),
                )
                if score:
                    scores.append((_host(target_logprob), _host(greedy), _host(greedy_logprob)))
            elif score:
                target_states, kcache, kpositions, vector, target_logprob, greedy, greedy_logprob = prefill_chunk_scored_program(
                    weights, kernel_weights, target_states, kcache, kpositions, vector,
                    put(padded[start:start + rows]), put(start).reshape(()), state_row, put(count).reshape(()),
                    put(padded[start + 1:start + rows + 1]),
                )
                scores.append((_host(target_logprob), _host(greedy), _host(greedy_logprob)))
            else:
                target_states, kcache, kpositions, vector = prefill_chunk_program(
                    weights, kernel_weights, target_states, kcache, kpositions, vector,
                    put(padded[start:start + rows]), put(start).reshape(()), state_row, put(count).reshape(()),
                )
            state_row = put(count).reshape(())
        if score:
            logprobs = np.concatenate([s[0] for s in scores])[: length - 1]
            greedy = np.concatenate([s[1] for s in scores])[: length - 1]
            greedy_logprobs = np.concatenate([s[2] for s in scores])[: length - 1]
            return target_states, state_row, kcache, kpositions, vector, (logprobs, greedy, greedy_logprobs)
        return target_states, state_row, kcache, kpositions, vector

    if args.target_only:
        def greedy_prefill_program(model_weights, target_states, prompt_tokens):
            token = jnp.zeros((target_batch_size,), jnp.int32)
            for index in range(prompt_tokens.shape[1]):
                target_states, token = greedy_token_program(
                    model_weights, target_states, put(prompt_tokens[:, index]),
                    put(np.full((target_batch_size,), index, np.int32)),
                )
            return target_states, token
    elif args.baseline:
        def greedy_prefill_program(model_weights, target_states, prompt_tokens):
            token = jnp.int32(0)
            for index in range(len(prompt_tokens)):  # one token per launch; prompt lengths vary
                target_states, token = greedy_token_program(
                    model_weights, target_states, put(int(prompt_tokens[index])).reshape(()), put(index).reshape(())
                )
            return target_states, token

    # First launches (loading the executables onto the devices; a compile when
    # the background compile could not be used).
    display.stage("first launches of the decode programs")
    phase_started = time.perf_counter()
    if not args.target_only:
        if "warm_states" in programs:
            warm_states = programs.pop("warm_states")
            warm_cache, warm_positions = programs.pop("warm_cache")
        else:
            warm_states = zero_states(rows)
            warm_cache, warm_positions = fused.empty_kernel_cache(c, mesh)
    display.note(f"state buffers ready {time.perf_counter() - phase_started:.1f} s")
    if args.baseline:
        # Launched first on purpose: shows whether the one-time first-execution
        # cost belongs to the runtime or to the verify program.
        phase_started = time.perf_counter()
        if args.target_only:
            greedy_states = programs.pop("warm_single", None) or zero_states(
                target_batch_size, sequence_rows=1
            )
            warm_tokens = put(np.full((target_batch_size,), 7, np.int32))
            warm_positions = put(np.zeros((target_batch_size,), np.int32))
        else:
            greedy_states = programs.pop("warm_single", None) or zero_states(1)
            warm_tokens = put(7).reshape(())
            warm_positions = put(0).reshape(())
        greedy_states, token = greedy_token_program(
            weights, greedy_states, warm_tokens, warm_positions
        )
        warm = greedy_decode_program(weights, greedy_states, token, warm_positions + 1)
        jax.block_until_ready(warm)
        launch_name = "target" if args.target_only else "baseline"
        display.note(f"{launch_name} launches {time.perf_counter() - phase_started:.1f} s")
        del warm, greedy_states, token
    if not args.target_only:
        phase_started = time.perf_counter()
        warm = prefill(warm_states, warm_cache, warm_positions, np.full((rows,), 7, np.int32))
        jax.block_until_ready(warm)
        display.note(f"prefill launch {time.perf_counter() - phase_started:.1f} s")
        phase_started = time.perf_counter()
        warm = speculate_program(weights, kernel_weights, *warm, put(rows).reshape(()))
        jax.block_until_ready(warm)
        display.note(f"speculative launch {time.perf_counter() - phase_started:.1f} s")
        del warm, warm_states, warm_cache, warm_positions
    display.finish(
        f"ready: target weights in {target_seconds:.0f} s on this host, total start-up "
        f"{time.perf_counter() - display.started:.0f} s (all hosts)"
        + (f" (background compile failed: {compile_error[0]})" if compile_error else "")
    )
    _log("")

    # --- prompt handling ----------------------------------------------------------
    def encode(prompt):
        if args.chat == "none":
            return np.asarray(tokenizer.encode("[BOS]" + prompt, allowed_special="all"), np.int32)
        return render_chat_tokens(tokenizer, prompt, args.chat)

    def read_prompt():
        """Process 0: the next prompt from --prompt, the inbox or the terminal (None ends the session)."""
        if args.prompt is not None:
            return args.prompt.pop(0) if args.prompt else None
        if args.inbox is not None:
            return inbox_prompt()
        while True:  # empty lines re-prompt; only quit/exit or a closed stdin end the session
            sys.stdout.write("prompt> ")
            sys.stdout.flush()
            line = prompt_stream.readline()
            if not line:
                import os
                try:
                    target = os.readlink(f"/proc/self/fd/{prompt_stream.fileno()}")
                except OSError as error:
                    target = f"unreadable ({error})"
                print(f"\nstdin is closed (-> {target}); srun did not forward the terminal. "
                      "Use --inbox FILE or --prompt for scripted runs. bye", flush=True)
                return None
            line = line.rstrip("\n")
            if line.strip().lower() in ("quit", "exit"):
                return None
            if line.strip():
                return line

    def next_job():
        """Process 0 obtains the next prompt (a terminal/inbox/--prompt line, or an HTTP request in
        server mode); every process receives the same tokens and generation parameters through one
        broadcast. Returns ``(tokens, max_tokens, stop_strings, score, ignore_eos, sampling, job)`` or
        None at the end of the session; ``sampling`` is None (greedy) or ``(temperature, top_p, seed)``;
        ``job`` is the server's request object on process 0 only."""
        buffer = np.full((JOB_HEADER + PROMPT_CAPACITY + STOP_CAPACITY,), -1, np.int32)
        job = None
        if jax.process_index() == 0:
            tokens, max_tokens, stop, score, ignore_eos = None, args.max_tokens, [], False, args.no_stop
            temperature, top_p, seed = args.temperature, args.top_p, args.seed
            if server is not None:
                try:
                    while True:
                        job = server.next_job(timeout=1.0)
                        if job is None:
                            continue
                        budget = args.context - len(job.tokens) - rows
                        stop_bytes = json.dumps(job.stop).encode("utf-8")
                        if budget < 0:
                            job.fail(f"prompt of {len(job.tokens)} tokens does not fit --context {args.context}")
                        elif len(stop_bytes) > STOP_CAPACITY:
                            job.fail("stop strings too long")
                        else:
                            tokens, max_tokens, stop, score = job.tokens, min(job.max_tokens, budget), job.stop, job.score
                            ignore_eos = job.ignore_eos or args.no_stop
                            temperature, top_p, seed = job.temperature, job.top_p, job.seed
                            break
                except KeyboardInterrupt:
                    job = None
            else:
                prompt = read_prompt()
                if prompt is not None:
                    tokens = encode(prompt)
                    if len(tokens) + args.max_tokens + rows > args.context:
                        print(f"prompt too long for --context {args.context} with --max-tokens {args.max_tokens}", flush=True)
                        tokens = tokens[: max(rows, args.context - args.max_tokens - rows)]
            if tokens is not None:
                stop_bytes = json.dumps(stop).encode("utf-8")
                buffer[0] = 1
                buffer[1] = len(tokens)
                buffer[2] = max_tokens
                buffer[3] = int(score)
                buffer[4] = len(stop_bytes)
                buffer[5] = int(ignore_eos)
                buffer[6] = int(temperature > 0)
                if seed is None:
                    seed = int(np.random.randint(0, 2**31 - 1))
                buffer[7:10] = openai_server.pack_sampling(temperature, top_p, seed)
                buffer[JOB_HEADER:JOB_HEADER + len(tokens)] = tokens
                buffer[JOB_HEADER + PROMPT_CAPACITY:JOB_HEADER + PROMPT_CAPACITY + len(stop_bytes)] = np.frombuffer(stop_bytes, np.uint8)
        buffer = np.asarray(mh.broadcast_one_to_all(buffer))
        if buffer[0] != 1:
            return None
        count, max_tokens, score, stop_length, ignore_eos, sampled = (int(v) for v in buffer[1:7])
        sampling = openai_server.unpack_sampling(*(int(v) for v in buffer[7:10])) if sampled else None
        tokens = buffer[JOB_HEADER:JOB_HEADER + count]
        stop_bytes = buffer[JOB_HEADER + PROMPT_CAPACITY:JOB_HEADER + PROMPT_CAPACITY + stop_length].astype(np.uint8).tobytes()
        return (tokens, max_tokens, json.loads(stop_bytes.decode("utf-8")) if stop_length else [], bool(score),
                bool(ignore_eos), sampling, job)

    inbox_state = {"offset": 0}

    def inbox_prompt():
        """Next non-empty line appended to the inbox file (blocks until one appears)."""
        path = args.inbox
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        if inbox_state["offset"] == 0:
            inbox_state["offset"] = path.stat().st_size  # ignore lines written before the demo was ready
            print(f"waiting for prompts appended to {path} (one per line; 'quit' ends)", flush=True)
        while True:
            with open(path, "r", errors="replace") as handle:
                handle.seek(inbox_state["offset"])
                line = handle.readline()
                if line.endswith("\n"):
                    inbox_state["offset"] = handle.tell()
                    text = line.strip()
                    if text.lower() in ("quit", "exit"):
                        return None
                    if text:
                        print(f"prompt> {text}", flush=True)
                        return text
                    continue
            time.sleep(0.5)

    class InteractiveTokenPrinter:
        """Print returned token blocks smoothly while the next device call runs.

        A multi-step device call returns all of its tokens together.  This
        background printer breaks that block into small groups and paces them
        over the duration of the call that produced it.  Generation can begin
        the next device call immediately instead of waiting for terminal I/O.
        """

        def __init__(self):
            self.items = queue.Queue()
            self.streamer = openai_server.TokenStreamer(
                tokenizer,
                chat=args.chat != "none",
                initial_channel="think" if args.chat == "think" else "response",
            )
            self.last_channel = None
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

        def submit(self, tokens, device_seconds=0.0):
            token_ids = [int(token) for token in tokens]
            if token_ids:
                self.items.put((token_ids, max(float(device_seconds), 0.0)))

        def close(self):
            self.items.put(None)
            self.thread.join()

        def _run(self):
            while True:
                item = self.items.get()
                if item is None:
                    return
                token_ids, device_seconds = item
                chunk_size = TERMINAL_STREAM_CHUNK_TOKENS
                chunk_count = math.ceil(len(token_ids) / chunk_size)
                delay = device_seconds / max(chunk_count, 1)
                for start in range(0, len(token_ids), chunk_size):
                    for channel, text in self.streamer.feed(token_ids[start:start + chunk_size]):
                        if channel == "content" and self.last_channel == "reasoning":
                            sys.stdout.write("\n\n[response]\n")
                        sys.stdout.write(text)
                        if text:
                            self.last_channel = channel
                    sys.stdout.flush()
                    if start + chunk_size < len(token_ids) and delay:
                        time.sleep(delay)

    prompt_stream = _prompt_stream()
    server = None
    if args.serve and jax.process_index() == 0:
        host, _, port = args.serve.rpartition(":")
        server = openai_server.OpenAIServer(
            host or "0.0.0.0", int(port), tokenizer, model_name=args.model_name,
            chat_mode="think" if args.chat == "think" else "response", default_max_tokens=args.max_tokens,
            max_prompt_tokens=min(PROMPT_CAPACITY, args.context - rows - 1),
            bos_token_id=int(tokenizer.encode("[BOS]", allowed_special="all")[0]), max_model_len=args.context,
            default_temperature=args.temperature, default_top_p=args.top_p, log=lambda m: _log(m),
        )
        server.start()
        _log("Ctrl-C ends the server session")
    elif jax.process_index() == 0 and args.prompt is None and args.inbox is None:
        if prompt_stream.isatty():
            # Discard keystrokes typed while the weights were loading.
            try:
                import termios
                termios.tcflush(prompt_stream.fileno(), termios.TCIFLUSH)
            except Exception:  # not a POSIX terminal
                pass
        print("Type a prompt and press Enter ('quit' to exit).", flush=True)
    interactive = server is None and jax.process_index() == 0

    def generate(prompt_tokens, max_tokens, stop, score, ignore_eos=False, sampling=None, job=None):
        """Prefill then speculate until an end token, a stop string, or ``max_tokens``.

        Every process runs this in lockstep and reaches the same decision (all hold the tokenizer
        and the same stop strings). With ``ignore_eos`` end tokens do not stop the generation.
        ``sampling`` is None (greedy) or ``(temperature, top_p, seed)`` for top-p sampling; the seed
        makes the generation reproducible and the device-side draws are identical on every host.
        ``job`` (process 0, server mode) receives the new tokens after every device call so a
        streaming client sees them as they are produced. Returns the generated ids (end token
        included when one ended the generation), the finish reason, the prompt scores
        (``score``), timings and step count."""
        length = len(prompt_tokens)
        if sampling is not None:
            ensure_sampled_programs()
        states = zero_states(rows)
        kcache, kpositions = fused.empty_kernel_cache(c, mesh)
        started = time.perf_counter()
        outputs = prefill(states, kcache, kpositions, prompt_tokens, score=score, sampling=sampling)
        states, state_row, kcache, kpositions, vector = outputs[:5]
        scores = outputs[5] if score else None
        jax.block_until_ready(vector)
        prefill_seconds = time.perf_counter() - started
        first_token = int(_host(vector)[0, fused.OUT_BONUS])

        generated = [first_token] if max_tokens > 0 else []
        accepted_counts = []
        call_seconds = []
        position = put(length).reshape(())
        printer = None
        if interactive:
            how = "" if sampling is None else f", sampling temperature {sampling[0]:g} top-p {sampling[1]:g} seed {sampling[2]}"
            print(f"\n[{length} prompt tokens, prefill {1e3 * prefill_seconds:.0f} ms{how}]\n", flush=True)
            printer = InteractiveTokenPrinter()
            printer.submit(generated)
        if sampling is None:
            def speculate_call(*state):
                return speculate_program(weights, kernel_weights, *state)
        else:
            sampling_dev = sampling_scalars(sampling)

            def speculate_call(*state):
                return speculate_sampled_program(weights, kernel_weights, *state, *sampling_dev)
        finish_reason = "length"
        finished = max_tokens == 0
        if generated and first_token in end_tokens and not ignore_eos:
            finished, finish_reason = True, "stop"
        if job is not None:
            job.emit(generated)

        def hit_stop_string():
            if not stop:
                return False
            text = tokenizer.decode(generated)
            return openai_server.apply_stop_strings(text, stop)[1]

        decode_started = time.perf_counter()
        while not finished and len(generated) < max_tokens:
            call_started = time.perf_counter()
            states, state_row, kcache, kpositions, vector, position, emitted, accepted = speculate_call(
                states, state_row, kcache, kpositions, vector, position
            )
            emitted_host = _host(emitted)
            call_elapsed = time.perf_counter() - call_started
            call_seconds.append(call_elapsed)
            accepted_counts.extend(int(a) for a in _host(accepted))
            before = len(generated)
            for token in emitted_host.reshape(-1):
                token = int(token)
                if token < 0:
                    continue
                generated.append(token)
                if token in end_tokens and not ignore_eos:
                    finished, finish_reason = True, "stop"
                    break
                if len(generated) >= max_tokens:
                    finished = True
                    break
            if not finished and hit_stop_string():
                finished, finish_reason = True, "stop"
            if job is not None:
                job.emit(generated[before:])
            if printer is not None:
                printer.submit(generated[before:], call_elapsed)
        decode_seconds = time.perf_counter() - decode_started
        steps = len(accepted_counts)
        if printer is not None:
            printer.close()
            sys.stdout.write("\n")
            sys.stdout.flush()
        if interactive:
            tokens_out = max(len(generated), 1)
            by_position = [float(np.mean([a > i for a in accepted_counts])) if steps else 0.0 for i in range(drafts)]
            print(f"\n--- {len(generated)} tokens in {decode_seconds:.2f} s: {len(generated) / max(decode_seconds, 1e-9):.1f} tokens/s, "
                  f"{1e3 * decode_seconds / tokens_out:.2f} ms/token")
            print(f"    {steps} speculative steps, {len(generated) / max(steps, 1):.2f} tokens/step, "
                  f"{1e3 * decode_seconds / max(steps, 1):.2f} ms/step; acceptance by draft position "
                  f"{[round(x, 2) for x in by_position]}")
            if call_seconds:
                print(f"    time to first token {1e3 * prefill_seconds:.0f} ms (prefill); median device call "
                      f"{1e3 * np.median(call_seconds):.1f} ms for {args.steps_per_call} steps")
        del states, kcache, kpositions
        return generated, finish_reason, scores, prefill_seconds, decode_seconds, steps

    def generate_target(job_specs):
        """Generate one independent completion per row of the target megakernel."""
        prompt_tokens = [spec[0] for spec in job_specs]
        max_tokens = [spec[1] for spec in job_specs]
        stop = [spec[2] for spec in job_specs]
        score = [spec[3] for spec in job_specs]
        ignore_eos = [spec[4] for spec in job_specs]
        jobs = [spec[6] for spec in job_specs]
        lengths = [len(tokens) for tokens in prompt_tokens]
        if any(score):
            raise ValueError("target-only generation does not support prompt scoring")
        if len(set(lengths)) != 1:
            raise ValueError("requests in a target batch must have equal prompt lengths")

        prompt_matrix = np.stack(prompt_tokens)
        states = zero_states(target_batch_size, sequence_rows=1)
        started = time.perf_counter()
        states, token = greedy_prefill_program(weights, states, prompt_matrix)
        jax.block_until_ready(token)
        prefill_seconds = time.perf_counter() - started
        first_tokens = [int(value) for value in _host(token).reshape(-1)]
        printer = None
        call_seconds = []

        generated = [
            [value] if budget > 0 else []
            for value, budget in zip(first_tokens, max_tokens)
        ]
        finish_reasons = ["length"] * target_batch_size
        finished = [budget == 0 for budget in max_tokens]
        if interactive:
            print(
                f"\n[{lengths[0]} prompt tokens, prefill {1e3 * prefill_seconds:.0f} ms]\n",
                flush=True,
            )
            printer = InteractiveTokenPrinter()
            printer.submit(generated[0])
        for row, value in enumerate(first_tokens):
            if not finished[row] and value in end_tokens and not ignore_eos[row]:
                finished[row] = True
                finish_reasons[row] = "stop"
            if jobs[row] is not None:
                jobs[row].emit(generated[row])

        position = put(np.asarray(lengths, np.int32))
        steps = 0
        decode_started = time.perf_counter()
        while not all(finished):
            call_started = time.perf_counter()
            states, token, output, position = greedy_decode_program(
                weights, states, token, position
            )
            output_host = _host(output)
            call_elapsed = time.perf_counter() - call_started
            call_seconds.append(call_elapsed)
            steps += output_host.shape[0]
            emitted = [[] for _ in range(target_batch_size)]
            for step_tokens in output_host:
                for row, raw_value in enumerate(step_tokens):
                    if finished[row]:
                        continue
                    value = int(raw_value)
                    generated[row].append(value)
                    emitted[row].append(value)
                    if value in end_tokens and not ignore_eos[row]:
                        finished[row] = True
                        finish_reasons[row] = "stop"
                    elif len(generated[row]) >= max_tokens[row]:
                        finished[row] = True
                    elif stop[row]:
                        decoded = tokenizer.decode(generated[row])
                        if openai_server.apply_stop_strings(decoded, stop[row])[1]:
                            finished[row] = True
                            finish_reasons[row] = "stop"
            for row, job in enumerate(jobs):
                if job is not None:
                    job.emit(emitted[row])
            if printer is not None:
                printer.submit(emitted[0], call_elapsed)
        decode_seconds = time.perf_counter() - decode_started
        if printer is not None:
            printer.close()
            sys.stdout.write("\n")
            sys.stdout.flush()
            tokens_out = max(len(generated[0]), 1)
            print(
                f"\n--- {len(generated[0])} tokens in {decode_seconds:.2f} s: "
                f"{len(generated[0]) / max(decode_seconds, 1e-9):.1f} tokens/s, "
                f"{1e3 * decode_seconds / tokens_out:.2f} ms/token"
            )
            if call_seconds:
                print(
                    f"    {steps} target steps; median device call "
                    f"{1e3 * np.median(call_seconds):.1f} ms for {args.steps_per_call} steps"
                )
        del states
        return generated, finish_reasons, prefill_seconds, decode_seconds, steps

    while True:
        if args.target_only:
            job_specs = []
            try:
                for _ in range(target_batch_size):
                    job_spec = next_job()
                    if job_spec is None:
                        break
                    job_specs.append(job_spec)
            except KeyboardInterrupt:
                break
            if not job_specs:
                break
            if len(job_specs) != target_batch_size:
                for spec in job_specs:
                    if spec[6] is not None:
                        spec[6].fail("server stopped before the target batch was full")
                break
            try:
                (
                    generated_rows,
                    finish_reasons,
                    prefill_seconds,
                    decode_seconds,
                    steps,
                ) = generate_target(job_specs)
            except Exception as error:  # noqa: BLE001 - report to every client in the batch
                for spec in job_specs:
                    if spec[6] is not None:
                        spec[6].fail(f"{type(error).__name__}: {error}")
                raise
            for row, spec in enumerate(job_specs):
                prompt_tokens, _, _, _, _, _, job = spec
                generated = generated_rows[row]
                if job is not None:
                    job.complete(openai_server.Result(
                        generated=generated, finish_reason=finish_reasons[row],
                        prefill_seconds=prefill_seconds, decode_seconds=decode_seconds, steps=steps,
                    ))
                    _log(
                        f"{job.kind}: {len(prompt_tokens)} prompt tokens "
                        f"(prefill {1e3 * prefill_seconds:.0f} ms), {len(generated)} generated in "
                        f"{decode_seconds:.6f} s ({steps} steps), finish {finish_reasons[row]}"
                    )
            _log(
                f"target_batch: {target_batch_size} requests, "
                f"{sum(len(values) for values in generated_rows)} generated in "
                f"{decode_seconds:.6f} s ({steps} steps)"
            )
            continue

        try:
            job_spec = next_job()
        except KeyboardInterrupt:  # the session was interrupted while waiting for a prompt or request
            break
        if job_spec is None:
            break
        prompt_tokens, max_tokens, stop, score, ignore_eos, sampling, job = job_spec
        length = len(prompt_tokens)
        try:
            generated, finish_reason, scores, prefill_seconds, decode_seconds, steps = generate(
                prompt_tokens, max_tokens, stop, score, ignore_eos, sampling, job
            )
        except Exception as error:  # noqa: BLE001 - report to the client, keep serving
            if job is not None:
                job.fail(f"{type(error).__name__}: {error}")
            raise
        if job is not None:
            result = openai_server.Result(
                generated=generated, finish_reason=finish_reason, prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds, steps=steps,
            )
            if scores is not None:
                result.prompt_logprobs = [float(v) for v in scores[0]]
                result.prompt_greedy = [int(v) for v in scores[1]]
                result.prompt_greedy_logprobs = [float(v) for v in scores[2]]
            job.complete(result)
            tags = ("scored" if score else None, f"sampled T={sampling[0]:g} p={sampling[1]:g}" if sampling else None)
            kind = job.kind + "".join(f" ({tag})" for tag in tags if tag)
            _log(f"{kind}: {length} prompt tokens (prefill {1e3 * prefill_seconds:.0f} ms), {len(generated)} generated in "
                 f"{decode_seconds:.6f} s ({steps} steps), finish {finish_reason}")
        if args.baseline and not args.target_only and server is None:
            tokens_out = max(len(generated), 1)
            states = zero_states(1)
            started = time.perf_counter()
            states, token = greedy_prefill_program(weights, states, prompt_tokens)
            jax.block_until_ready(token)
            baseline_prefill = time.perf_counter() - started
            baseline_tokens = [int(_host(token))]
            position_b = put(length).reshape(())
            started = time.perf_counter()
            while len(baseline_tokens) < len(generated):
                states, token, out, position_b = greedy_decode_program(
                    weights, states, token, position_b
                )
                out_host = _host(out)
                baseline_tokens.extend(int(t) for t in out_host)
            jax.block_until_ready(token)
            baseline_seconds = time.perf_counter() - started
            baseline_seconds *= len(generated) / len(baseline_tokens)  # the last call overshoots the budget
            baseline_tokens = baseline_tokens[:len(generated)]
            if jax.process_index() == 0:
                agree = 0
                while agree < min(len(baseline_tokens), len(generated)) and baseline_tokens[agree] == generated[agree]:
                    agree += 1
                print(f"    B1 greedy baseline: {len(baseline_tokens)} tokens in {baseline_seconds:.2f} s "
                      f"({1e3 * baseline_seconds / len(baseline_tokens):.2f} ms/token, prefill {1e3 * baseline_prefill:.0f} ms); "
                      f"speedup {(baseline_seconds / len(baseline_tokens)) / (max(decode_seconds, 1e-9) / tokens_out):.2f}x; "
                      f"agrees on the first {agree} tokens")
            del states
        if interactive:
            print(flush=True)
    if server is not None:
        server.shutdown()
    _log("bye")


if __name__ == "__main__":
    main()
