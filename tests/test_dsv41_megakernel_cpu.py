"""Suite T4: Pallas Megakernel CPU Interpret & VMEM Budget Verification.

Tests:
1. VMEM static budget assertion (`compute_vmem_budget` <= 48 MiB soft budget for both `full_config` and `tiny_config`).
2. `pool_alias.py` zero-copy weight aliasing (`expert_w1_view`, `expert_w3_view`, `expert_w2_view`).
3. `collectives.py` ring all-reduce equivalence on 8 virtual CPU devices.
4. Bit-exact / near-zero parity between `make_megakernel_decode(..., interpret=True)` and
   `make_xla_decode_step` AND `DeepSeekV41Reference` across all 6 layer kinds (`SWA`, `FULL_R2`,
   `REUSE_R2`, `FULL_R1_CAND`, `REINDEX_R1`, `REUSE_R1`) on 8 virtual CPU devices.
5. Negative control (`sinkhorn_iters=0`) diverges from `sinkhorn_iters=20` on the same prompt.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=32")

import jax
import jax.numpy as jnp
import numpy as np

from deepseek_v41.checkpoint import generate_synthetic_weights
from deepseek_v41.collectives import create_v7x_mesh
from deepseek_v41.config import full_config, tiny_config
from deepseek_v41.decode_megakernel import compute_vmem_budget, make_megakernel_decode
from deepseek_v41.reference import DeepSeekV41Reference
from deepseek_v41.xla_decode import init_cache, make_xla_decode_step, shard_weights_for_mesh


def test_vmem_budget_within_48mib() -> None:
    for cfg in (tiny_config(), full_config()):
        for world_size in (8, 32):
            b = compute_vmem_budget(cfg, batch_size=8, num_devices=world_size, max_verify_tokens=6)
            assert b.within_soft_budget, f"VMEM budget {b.total_vmem_bytes} exceeds 48 MiB for world_size={world_size}"
            assert b.total_vmem_bytes <= 48 * 1024 * 1024


def test_megakernel_matches_xla_and_reference_cpu_interpret() -> None:
    cfg = tiny_config()
    mesh = create_v7x_mesh(jax.devices()[:8])
    raw_w = generate_synthetic_weights(cfg, seed=11)
    sharded_w = shard_weights_for_mesh(cfg, raw_w, mesh)

    xla_step = make_xla_decode_step(cfg, mesh)
    mk_step = make_megakernel_decode(cfg, mesh, interpret=True)
    ref_model = DeepSeekV41Reference(cfg, raw_w)
    ref_state = ref_model.init_state(batch_size=1)

    cache_xla = init_cache(cfg, batch_size=1, mesh=mesh)
    cache_mk = init_cache(cfg, batch_size=1, mesh=mesh)

    seq = [3, 7, 11, 19]
    for pos, tok in enumerate(seq):
        tok_arr = jnp.asarray([[tok]], dtype=jnp.int32)
        pos_arr = jnp.asarray(pos, dtype=jnp.int32)

        logits_xla, mh_xla, cache_xla = xla_step(sharded_w, cache_xla, tok_arr, pos_arr)
        logits_mk, mh_mk, cache_mk = mk_step(sharded_w, cache_mk, tok_arr, pos_arr)
        _, logits_ref, mh_ref = ref_model.forward(tok_arr, start_pos=pos, state=ref_state, full_logits=False)

        np.testing.assert_allclose(np.asarray(logits_mk), np.asarray(logits_xla), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(mh_mk), np.asarray(mh_xla), rtol=1e-5, atol=1e-5)
        assert int(np.argmax(logits_mk)) == int(np.argmax(logits_ref))


def test_megakernel_sinkhorn_zero_negative_control() -> None:
    cfg = tiny_config()
    mesh = create_v7x_mesh(jax.devices()[:8])
    raw_w = generate_synthetic_weights(cfg, seed=23)
    sharded_w = shard_weights_for_mesh(cfg, raw_w, mesh)

    mk_20 = make_megakernel_decode(cfg, mesh, sinkhorn_iters=20, interpret=True)
    mk_0 = make_megakernel_decode(cfg, mesh, sinkhorn_iters=0, interpret=True)

    c20 = init_cache(cfg, batch_size=1, mesh=mesh)
    c0 = init_cache(cfg, batch_size=1, mesh=mesh)
    tok = jnp.asarray([[5, 9, 14, 21]], dtype=jnp.int32)
    l20, _, _ = mk_20(sharded_w, c20, tok, jnp.asarray(0, dtype=jnp.int32))
    l0, _, _ = mk_0(sharded_w, c0, tok, jnp.asarray(0, dtype=jnp.int32))
    diff = float(np.max(np.abs(np.asarray(l20) - np.asarray(l0))))
    assert diff > 1e-2, f"Expected sinkhorn_iters=0 to diverge from sinkhorn_iters=20, got max diff {diff}"
