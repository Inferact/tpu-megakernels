"""Regression test of the XLA prefill on the REAL Muse Spark 1.2 int4 weights.

Re-runs the first prompt of `scripts/validate_musespark_prefill.py` (the chat-rendered
"What is 2+2?") with prefill-as-decode and checks that the greedy continuation and the
softcapped logits after the prompt equal the saved oracle. Skipped unless the pre-sharded
container, the oracle directory and eight TPU devices are present. Loads ~440 GB of weights,
so it takes several minutes; run it alone through the TPU launcher:

    XLA_FLAGS=--xla_allow_excess_precision=false tpu_run.sh all \
        python -m pytest tests/test_musespark_real_prefill.py -s

`MUSESPARK_REAL_REF` / `MUSESPARK_WEIGHTS` / `MUSESPARK_CHECKPOINT` override the locations.
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_musespark_prefill.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("validate_musespark_prefill", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V = _load_script()
WEIGHTS = Path(os.environ.get("MUSESPARK_WEIGHTS", V.DEFAULT_WEIGHTS))
CHECKPOINT = Path(os.environ.get("MUSESPARK_CHECKPOINT", V.DEFAULT_CHECKPOINT))
REF = Path(os.environ.get("MUSESPARK_REAL_REF", V.DEFAULT_OUT))
PROMPT = 0
# The same executable (compilation cache) reproduces the oracle bit-exactly; a different XLA
# build of the same program was measured 0.38 apart after 62 layers of bf16 noise (the
# taps=True program against the taps=False one), so the logits are checked to 0.5 and the
# argmax / greedy continuation exactly.
LOGIT_TOL = 0.5


def _needs():
    """Reason to skip, or None."""
    for name in ("ids", "logits", "gen"):
        if not (REF / f"prompt{PROMPT}_{name}.npy").is_file():
            return f"oracle {REF}/prompt{PROMPT}_{name}.npy missing (run {SCRIPT.name})"
    if not (WEIGHTS / "layout.json").is_file():
        return f"container {WEIGHTS} missing"
    if not (CHECKPOINT / "tokenizer.json").is_file():
        return f"checkpoint {CHECKPOINT} missing"
    try:
        import jax

        devices = jax.devices()
    except Exception as error:  # noqa: BLE001
        return f"no JAX devices: {error}"
    if len(devices) < V.TP or devices[0].platform != "tpu":
        return f"needs {V.TP} TPU devices, found {[d.platform for d in devices]}"
    return None


@pytest.mark.skipif(_needs() is not None, reason=_needs() or "")
def test_real_prefill_matches_saved_oracle(capsys):
    import jax

    from musespark import load as ms_load

    ref_ids = np.load(REF / f"prompt{PROMPT}_ids.npy")
    ref_logits = np.load(REF / f"prompt{PROMPT}_logits.npy")
    ref_gen = np.load(REF / f"prompt{PROMPT}_gen.npy")
    summary = json.loads((REF / "summary.json").read_text())
    context = int(summary.get("context", 4096))

    doc = ms_load.read_layout(WEIGHTS)
    cfg = ms_load.config_from_layout(doc)
    tokenizer = ms_load.load_tokenizer(CHECKPOINT)
    name, kind, text = V.prompt_specs()[PROMPT]
    ids, _ = V.encode_prompt(cfg, tokenizer, kind, text)
    assert ids == ref_ids.tolist(), "the rendered prompt changed (tokenizer / chat template)"
    assert ref_logits.shape == (cfg.vocab,)

    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[: V.TP]), ("tp",))
    started = time.perf_counter()
    weights = ms_load.load_presharded(mesh, WEIGHTS, cfg, log=lambda _: None)
    log(f"weights resident in {time.perf_counter() - started:.0f} s")
    runner = V.Runner(mesh, cfg, weights, context, log)
    generated, logits, times, reason = runner.generate(
        ids, len(ref_gen), deadline=time.perf_counter() + 15 * 60
    )
    log(f"{len(generated)} tokens ({reason}) in {sum(times):.1f} s: {tokenizer.decode(generated)!r}")

    diff = np.abs(logits - ref_logits)
    log(f"logits max |diff| {diff.max():.3e}, exact {np.mean(diff == 0):.4f}")
    assert np.all(np.abs(logits) <= cfg.softcap)
    assert diff.max() <= LOGIT_TOL
    assert int(np.argmax(logits[: cfg.vocab_used])) == int(np.argmax(ref_logits[: cfg.vocab_used]))
    assert reason != "deadline"
    assert generated == ref_gen.tolist(), (
        f"continuation changed:\n got {tokenizer.decode(generated)!r}\n ref {tokenizer.decode(ref_gen)!r}"
    )
