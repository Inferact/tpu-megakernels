"""TPU v7x mesh builder and Pallas in-kernel collective helpers for DeepSeek-V4.1-Flash.

Supports 32-device four-host (`TP32`) two-stage XOR tree collectives via `collectives32`,
as well as smaller power-of-two meshes for local CPU / unit testing.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
import os
from typing import Sequence

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

import collectives32


def scope(name: str):
    """Optional device trace scope enabled by TP32_PROFILE_SCOPES=1."""
    return (
        jax.named_scope("dsv41_tp/" + name)
        if os.environ.get("TP32_PROFILE_SCOPES") == "1"
        else nullcontext()
    )


def _permute_order(num_devices: int, device_order_bits: str | None) -> tuple[tuple[int, ...], list[int]]:
    """Compute physical device index permutation preserving chip-pair bit 0 (`rank ^ 1`)."""
    n_bits = int(round(math.log2(num_devices)))
    if (1 << n_bits) != num_devices:
        raise ValueError(f"num_devices must be a power of 2, got {num_devices}")

    if device_order_bits is None:
        bits = tuple(range(n_bits))
    else:
        parsed = tuple(int(x.strip()) for x in device_order_bits.split(",") if x.strip() != "")
        if len(parsed) == n_bits:
            bits = parsed
        elif len(parsed) == 5 and n_bits <= 5:
            # Project a 5-bit TP32 order down to the active low `n_bits` while preserving relative order.
            low_bits = [b for b in parsed if b < n_bits]
            bits = tuple(low_bits)
        elif len(parsed) == 3 and n_bits == 5:
            # Apply 3-bit intra-host permutation within each 8-device host.
            bits = tuple(parsed) + (3, 4)
        else:
            raise ValueError(
                f"device_order_bits={device_order_bits!r} incompatible with num_devices={num_devices}"
            )

    if sorted(bits) != list(range(n_bits)) or (n_bits > 0 and bits[0] != 0):
        raise ValueError(
            f"device_order_bits must permute 0..{n_bits - 1} while preserving chip-pair bit 0; got {bits}"
        )

    order = [
        sum(((rank >> bit) & 1) << physical for bit, physical in enumerate(bits))
        for rank in range(num_devices)
    ]
    return bits, order


def create_v7x_mesh(
    num_devices: int | None = None,
    *,
    device_order_bits: str | None = None,
    axis_name: str = "tp",
    devices: Sequence[jax.Device] | None = None,
    return_order: bool = False,
):
    """Build a 1-D v7x mesh sorted by `(process_index, id)` with `device_order_bits` permutation.

    Within every 8-device host, bit 0 is preserved so `rank ^ 1` is always the on-chip sibling
    TensorCore. Supports `num_devices` in `{2, 8, 32}` (and any power-of-2 slice for CPU tests).
    """
    if num_devices is not None and not isinstance(num_devices, int):
        devices = num_devices
        num_devices = len(devices)
    all_devices = sorted(
        list(devices) if devices is not None else jax.devices(),
        key=lambda d: (d.process_index, d.id),
    )
    if num_devices is None:
        num_devices = len(all_devices)
    if len(all_devices) < num_devices:
        raise RuntimeError(
            f"Requested {num_devices} devices for mesh '{axis_name}', but only {len(all_devices)} available"
        )

    selected = all_devices[:num_devices]
    bits, order = _permute_order(num_devices, device_order_bits)
    ordered_devices = np.array([selected[idx] for idx in order], dtype=object)
    mesh = Mesh(ordered_devices, (axis_name,))
    if return_order:
        return mesh, bits, order
    return mesh


def _cpu_pad_barrier(x):
    return jax.lax.optimization_barrier(x) if jax.default_backend() == "cpu" else x


def collective_scratch(num_devices: int = 32):
    """Allocate Pallas VMEM + DMA semaphore scratch shapes for in-kernel XOR reductions (`N in {2, 8, 32}`)."""
    host_lanes = min(num_devices, 8)
    num_hosts = max(1, num_devices // 8)
    peers = max(1, (host_lanes - 1) + (num_hosts - 1))
    return (
        pltpu.VMEM((2, host_lanes, 320, 128), jnp.float32),
        pltpu.VMEM((2, num_hosts, 320, 128), jnp.float32),
        pltpu.SemaphoreType.DMA((peers,)),
        pltpu.SemaphoreType.DMA((peers,)),
    )


def barrier(
    num_devices: int = 32,
    *,
    axis_name: str = "tp",
    interpret: bool = False,
) -> None:
    """Synchronize all ranks in `axis_name` using barrier semaphores on TPU or no-op in interpret."""
    if interpret or num_devices <= 1:
        return
    rank = jax.lax.axis_index(axis_name)
    sem = pltpu.get_barrier_semaphore()
    for offset in range(1, num_devices):
        pl.semaphore_signal(
            sem,
            1,
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(sem, num_devices - 1)


def intra_host_all_reduce(
    x: jax.Array,
    *,
    local_vmem=None,
    sends=None,
    recvs=None,
    slot: int = 0,
    num_devices: int = 32,
    axis_name: str = "tp",
    interpret: bool = False,
) -> jax.Array:
    """All-reduce `x` in FP32 across the `min(num_devices, 8)` lanes of the current host (`rank ^ 0..7`)."""
    host_lanes = min(num_devices, 8)
    x_f32 = x.astype(jnp.float32)
    if host_lanes <= 1:
        return x_f32

    n_elems = int(np.prod(x_f32.shape))
    rows_128 = (n_elems + 127) // 128
    rows_pad = ((max(rows_128, 8) + 7) // 8) * 8

    if (
        interpret
        or local_vmem is None
        or sends is None
        or recvs is None
        or rows_pad > local_vmem.shape[2]
    ):
        if num_devices <= 8:
            return jax.lax.psum(x_f32, axis_name)
        rank = jax.lax.axis_index(axis_name)
        host_base = (rank // 8) * 8
        gathered = jax.lax.all_gather(x_f32, axis_name, axis=0)
        host_slice = jax.lax.dynamic_slice_in_dim(gathered, host_base, 8, axis=0)
        return jnp.sum(host_slice, axis=0)

    rank = jax.lax.axis_index(axis_name)
    s = slot % 2
    flat_1d = x_f32.reshape(-1)
    pad_elems = rows_pad * 128 - n_elems
    if pad_elems > 0:
        flat_1d = jnp.pad(flat_1d, (0, pad_elems))
    flat_2d = flat_1d.reshape(rows_pad, 128)

    barrier(num_devices, axis_name=axis_name, interpret=False)
    local_vmem[s, 0, :rows_pad, :] = flat_2d
    transfers = []
    for offset in range(1, host_lanes):
        dma = pltpu.make_async_remote_copy(
            local_vmem.at[s, 0, :rows_pad, :],
            local_vmem.at[s, offset, :rows_pad, :],
            sends.at[offset - 1],
            recvs.at[offset - 1],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        dma.start()
        transfers.append(dma)
    for dma in transfers:
        dma.wait()

    total_2d = local_vmem[s, 0, :rows_pad, :]
    for offset in range(1, host_lanes):
        total_2d = total_2d + local_vmem[s, offset, :rows_pad, :]
    return total_2d.reshape(-1)[:n_elems].reshape(x.shape)


def all_reduce_sum(
    x: jax.Array,
    local_vmem=None,
    hosts_vmem=None,
    sends=None,
    recvs=None,
    *,
    output_ref=None,
    iteration: int = 0,
    num_devices: int = 32,
    axis_name: str = "tp",
    interpret: bool = False,
) -> jax.Array:
    """All-reduce `x` across all `num_devices` ranks (`N in {2, 8, 32}`) in FP32."""
    x_f32 = x.astype(jnp.float32)
    if num_devices <= 1:
        if output_ref is not None:
            output_ref[...] = x_f32.astype(output_ref.dtype)
        return x_f32

    n_elems = int(np.prod(x_f32.shape))
    rows_128 = (n_elems + 127) // 128
    rows_pad = ((max(rows_128, 8) + 7) // 8) * 8

    use_hardware_tree = (
        not interpret
        and local_vmem is not None
        and sends is not None
        and recvs is not None
        and rows_pad <= local_vmem.shape[2]
    )
    if not use_hardware_tree:
        orig_shape = x_f32.shape
        d = orig_shape[-1]
        x2d = x_f32.reshape(-1, d)
        m = x2d.shape[0]
        pad_m = ((max(m, 8) + 7) // 8) * 8
        if pad_m > m:
            x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
        x2d = _cpu_pad_barrier(x2d)
        total2d = jax.lax.psum(x2d, axis_name)
        total2d = _cpu_pad_barrier(total2d)
        total = total2d[:m].reshape(orig_shape)
        if output_ref is not None:
            output_ref[...] = total.astype(output_ref.dtype)
        return total

    # Stage 1: Intra-host 8-lane (or 2-lane) XOR reduction across rank ^ 1..min(num_devices, 8)-1
    intra_total = intra_host_all_reduce(
        x_f32,
        local_vmem=local_vmem,
        sends=sends,
        recvs=recvs,
        slot=iteration,
        num_devices=num_devices,
        axis_name=axis_name,
        interpret=False,
    )

    num_hosts = max(1, num_devices // 8)
    if num_hosts <= 1 or hosts_vmem is None:
        if output_ref is not None:
            output_ref[...] = intra_total.astype(output_ref.dtype)
        return intra_total

    # Stage 2 (N = 32): Inter-host 4-way XOR reduction across rank ^ 8, rank ^ 16, rank ^ 24
    rank = jax.lax.axis_index(axis_name)
    s = iteration % 2
    host_lanes = min(num_devices, 8)
    sem_base = host_lanes - 1
    flat_1d = intra_total.reshape(-1)
    pad_elems = rows_pad * 128 - n_elems
    if pad_elems > 0:
        flat_1d = jnp.pad(flat_1d, (0, pad_elems))
    hosts_vmem[s, 0, :rows_pad, :] = flat_1d.reshape(rows_pad, 128)

    host_transfers = []
    for h_off in range(1, num_hosts):
        dma = pltpu.make_async_remote_copy(
            hosts_vmem.at[s, 0, :rows_pad, :],
            hosts_vmem.at[s, h_off, :rows_pad, :],
            sends.at[sem_base + h_off - 1],
            recvs.at[sem_base + h_off - 1],
            device_id=(rank ^ (h_off * 8),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        dma.start()
        host_transfers.append(dma)
    for dma in host_transfers:
        dma.wait()

    total_2d = hosts_vmem[s, 0, :rows_pad, :]
    for h_off in range(1, num_hosts):
        total_2d = total_2d + hosts_vmem[s, h_off, :rows_pad, :]
    total = total_2d.reshape(-1)[:n_elems].reshape(x.shape)
    if output_ref is not None:
        output_ref[...] = total.astype(output_ref.dtype)
    return total
