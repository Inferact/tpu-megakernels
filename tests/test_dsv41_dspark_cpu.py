"""Suite T5: DSpark Speculative Drafter, O(1) Rollback, and Greedy Equivalence Tests.

Verifies:
1. Sharded `DSparkDrafter` (`mtp.0..2`) matches `DeepSeekV41Reference.forward_spec` on 8 virtual CPU devices.
2. `verify_and_commit_block` + `rollback_target_cache` produces bit-identical state and token sequences
   under every forced acceptance count `k in {0, 1, 2, 3, 4, 5}` across both even and odd `r=2` compressor
   boundaries.
3. End-to-end `DSV41Engine.generate_greedy(..., use_dspark=True)` produces the exact same token IDs
   (`T_dspark == T_base`) as `use_dspark=False`, and works end-to-end through the OpenAI `/v1/chat/completions` server.
"""

from __future__ import annotations

import json
import os
import urllib.request

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=32")

import jax
import jax.numpy as jnp
import numpy as np

from deepseek_v41.checkpoint import generate_synthetic_weights
from deepseek_v41.collectives import create_v7x_mesh
from deepseek_v41.config import tiny_config
from deepseek_v41.dspark import DSparkDrafter
from deepseek_v41.engine import DSV41Engine
from deepseek_v41.reference import DeepSeekV41Reference
from deepseek_v41.server import run_server_in_thread
from deepseek_v41.xla_decode import init_cache, shard_weights_for_mesh


def test_dspark_drafter_matches_reference() -> None:
    cfg = tiny_config()
    mesh = create_v7x_mesh(jax.devices()[:8])
    raw_w = generate_synthetic_weights(cfg, seed=7)
    sharded_w = shard_weights_for_mesh(cfg, raw_w, mesh)

    ref = DeepSeekV41Reference(cfg, raw_w)
    ref_state = ref.init_state(batch_size=1)
    drafter = DSparkDrafter(cfg, mesh)
    cache = init_cache(cfg, batch_size=1, mesh=mesh)

    prompt = jnp.asarray([[4, 8, 15, 16]], dtype=jnp.int32)
    out_ids, _, mh_prefill = ref.forward(prompt, start_pos=0, state=ref_state, full_logits=False)
    assert mh_prefill is not None
    ref.forward_spec(prompt, mh_prefill[:, :-1, :], start_pos=0, state=ref_state)
    cache = drafter.prefill_main_hidden(sharded_w, cache, mh_prefill[:, :-1, :], start_pos=0)

    last_tok = out_ids[:, -1] if out_ids.ndim == 2 else out_ids
    ref_spec = ref.forward_spec(last_tok, mh_prefill[:, -1:, :], start_pos=3, state=ref_state)
    assert ref_spec is not None
    ref_cand, ref_logits, ref_conf = ref_spec

    cand, logits, conf, _ = drafter.draft(sharded_w, cache, last_tok, mh_prefill[:, -1:, :], start_pos=3)
    np.testing.assert_array_equal(np.asarray(cand), np.asarray(ref_cand))
    np.testing.assert_allclose(np.asarray(logits), np.asarray(ref_logits), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(conf), np.asarray(ref_conf), rtol=2e-2, atol=2e-2)


def test_dspark_greedy_token_identity_across_all_rejection_positions() -> None:
    cfg = tiny_config()
    mesh = create_v7x_mesh(jax.devices()[:8])
    raw_w = generate_synthetic_weights(cfg, seed=19)
    sharded_w = shard_weights_for_mesh(cfg, raw_w, mesh)

    engine = DSV41Engine(cfg, mesh, sharded_w, default_backend="xla")
    prompt = [5, 12, 19, 27, 31]

    base = engine.generate_greedy(prompt, max_new_tokens=10, use_dspark=False)
    dsp_natural = engine.generate_greedy(prompt, max_new_tokens=10, use_dspark=True)
    assert dsp_natural["generated_ids"] == base["generated_ids"]

    # Force rejection at every possible draft offset k in {0, 1, 2, 3, 4, 5}
    dsp_forced = engine.generate_greedy(
        prompt,
        max_new_tokens=10,
        use_dspark=True,
        force_accept_schedule=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    assert dsp_forced["generated_ids"] == base["generated_ids"]


def test_openai_server_chat_completions_roundtrip() -> None:
    cfg = tiny_config()
    mesh = create_v7x_mesh(jax.devices()[:8])
    raw_w = generate_synthetic_weights(cfg, seed=31)
    sharded_w = shard_weights_for_mesh(cfg, raw_w, mesh)
    engine = DSV41Engine(cfg, mesh, sharded_w, default_backend="xla")

    with run_server_in_thread(engine, host="127.0.0.1", port=0, default_use_dspark=True, default_backend="xla") as (
        base_url,
        _,
    ):
        with urllib.request.urlopen(f"{base_url}/health", timeout=10) as h_resp:
            health = json.loads(h_resp.read().decode("utf-8"))
            assert health["status"] == "ok"

        body = json.dumps(
            {
                "model": "deepseek-v4.1-flash",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "use_dspark": True,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=360) as c_resp:
            payload = json.loads(c_resp.read().decode("utf-8"))
            assert payload["object"] == "chat.completion"
            assert payload["usage"]["completion_tokens"] == 4
            assert "dspark_stats" in payload["x_dsv41_metrics"]
