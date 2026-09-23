"""TP32 reduction: eight ranks per host, then four hosts.

O/down projections issue two disjoint local packets before reduce() waits for
both. Slots alternate between reductions. Source slot zero stays intact until
all send acknowledgments complete; local and host phases use distinct semaphores.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from contextlib import nullcontext
import os


def scope(name):
    """Optional device trace labels; enabled by TP32_PROFILE_SCOPES=1."""
    return (
        jax.named_scope("tp32/" + name)
        if os.environ.get("TP32_PROFILE_SCOPES") == "1"
        else nullcontext()
    )


def barrier():
    rank = jax.lax.axis_index("tp")
    sem = tpu.get_barrier_semaphore()
    for offset in range(1, 32):
        pl.semaphore_signal(sem, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(sem, 31)


def start_local_piece(x, local, sends, recvs, iteration, piece):
    """Send one 4096-element projection half to the other seven local ranks."""
    rank = jax.lax.axis_index("tp")
    slot = iteration % 2
    region = pl.ds(piece * 32, 32)
    local[slot, 0, region, :] = x[...].reshape(32, 128)
    for offset in range(7, 0, -1):
        tpu.make_async_remote_copy(
            local.at[slot, 0, region, :],
            local.at[slot, offset, region, :],
            sends.at[offset - 1],
            recvs.at[offset - 1],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()


def encode_partial(x, packet):
    """Round a host sum to signed int16 with a shared power-of-two scale."""
    maximum = jnp.max(jnp.abs(x[...]))
    exponent = jnp.maximum((jax.lax.bitcast_convert_type(maximum, jnp.uint32) >> 23) & 255, 15) - 14
    inverse_scale = jax.lax.bitcast_convert_type((jnp.uint32(254) - exponent) << 23, jnp.float32)
    quantized = jnp.clip(jnp.rint(x[...] * inverse_scale), -32767, 32767).astype(jnp.int16)
    packet[:64, :] = jax.lax.bitcast_convert_type(quantized, jnp.uint16).reshape(64, 128)
    packet[64, :] = jnp.broadcast_to(exponent.astype(jnp.uint16), (128,))
    packet[65, :] = jnp.zeros((128,), jnp.uint16)


def reduce(output, local, hosts, sends, recvs, iteration):
    """Finish the local gather, exchange host sums, and return an FP32 sum."""
    rank = jax.lax.axis_index("tp")
    slot = iteration % 2
    with scope("local_wait"):
        for offset in range(7, 0, -1):
            # A full-size wait consumes both half-packet completions.
            tpu.make_async_remote_copy(
                local.at[slot, 0],
                local.at[slot, offset],
                sends.at[offset - 1],
                recvs.at[offset - 1],
                device_id=(rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            ).wait()
    with scope("local_sum"):
        output[...] = jnp.sum(local[slot, ...].astype(jnp.float32), axis=0).reshape(output.shape)
    with scope("encode"):
        encode_partial(output, hosts.at[slot, 0])
    transfers = []
    with scope("remote_issue"):
        for offset in range(3, 0, -1):
            dma = tpu.make_async_remote_copy(
                hosts.at[slot, 0],
                hosts.at[slot, offset],
                sends.at[6 + offset],
                recvs.at[6 + offset],
                device_id=(rank ^ (offset * 8),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            dma.start()
            transfers.append(dma)
    with scope("remote_wait"):
        for dma in transfers:
            dma.wait()
    with scope("decode_sum"):
        data = hosts[slot, ...]
        scales = jax.lax.bitcast_convert_type(
            data[..., 64:65, :].astype(jnp.uint32) << 23, jnp.float32
        )
        values = jax.lax.bitcast_convert_type(data[..., :64, :], jnp.int16).astype(jnp.float32)
        output[...] = jnp.sum(values * scales, axis=0).reshape(output.shape)


def scratch():
    return (
        tpu.VMEM((2, 8, 64, 128), jnp.bfloat16),
        tpu.VMEM((2, 4, 66, 128), jnp.uint16),
        tpu.SemaphoreType.DMA((10,)),
        tpu.SemaphoreType.DMA((10,)),
    )
