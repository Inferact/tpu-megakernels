"""Kimi K3 decode megakernel components."""

from __future__ import annotations

import dataclasses
import functools
import math
from types import SimpleNamespace
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import typing as jax_typing
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

import collectives32
import kimi
import pool_alias


_COMMUNICATION_SCRATCH_SPEC_COUNT = 10


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class _PaddedBf16Rows(pool_alias.Reinterpret):
    """Expose active rows while preserving the full native 16-row capacity."""

    def _validate_shape(self, shape):
        rows, width = self.shape
        assert len(self.shape) == 2 and 1 <= rows <= 16 and width % 128 == 0
        assert math.prod(shape) == 16 * width, (shape, self.shape)


def _bf16_rows_view(ref, rows, width):
    ref = ref.bitcast(jnp.bfloat16)
    return pool_alias.types.TransformedRef(
        ref.ref,
        (*ref.transforms, _PaddedBf16Rows((rows, width))),
    )


def _dot(x, w):
    batch_size = x.shape[0]
    if not 1 <= batch_size <= 16:
        raise ValueError("The native MXU row tile supports batch sizes from 1 to 16")
    if batch_size == 1:
        # Preserve the established batch-one lowering and numerics exactly.
        tiled_x = jnp.broadcast_to(x, (16, x.shape[-1]))
    else:
        tiled_x = jnp.pad(x, ((0, 16 - batch_size), (0, 0)))
    return jax.lax.dot_general(
        tiled_x,
        w,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )[:batch_size]


def _attention_scores(queries, keys):
    """Multiply partial query rows by a transposed key tile in FP32."""
    row_count = queries.shape[0]
    padded_queries = jnp.pad(queries, ((0, 16 - row_count), (0, 0)))
    return jax.lax.dot_general(
        padded_queries,
        keys,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )[:row_count]


def _probability_value_mxu(
    probabilities,
    values,
    mode: Literal["bf16", "hilo"],
):
    """Multiply attention probabilities by a BF16 value tile on the MXU.

    ``hilo`` represents each FP32 probability as the sum of two BF16 values,
    retaining roughly 16 mantissa bits across two MXU passes.
    """
    row_count = probabilities.shape[0]
    padded_row_count = max(16 - row_count, 0)

    def dot(left):
        return jax.lax.dot_general(
            jnp.pad(left, ((0, padded_row_count), (0, 0))),
            values,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )[:row_count]

    high = probabilities.astype(jnp.bfloat16)
    result = dot(high)
    if mode == "hilo":
        low = (probabilities - high.astype(jnp.float32)).astype(jnp.bfloat16)
        result += dot(low)
    return result


def _gated_activation(gate, value):
    """Kimi K3 shared/routed expert activation, rounded to BF16."""
    return (
        4
        * jnp.tanh(gate / 4)
        * jax.nn.sigmoid(gate)
        * 25
        * jnp.tanh(value / 25)
    ).astype(jnp.bfloat16)
def _route_hidden_states(
    hidden_states,
    routing_projection_vmem,
    routing_bias_ref,
    routing_scores_vmem,
    send_semaphore,
    receive_semaphore,
    *,
    selected_expert_count,
    local_expert_count,
    expert_tensor_parallel,
    on_local_selection,
    after_selection,
):
    """Project and select local expert routes through one shared B1/B2 path.

    ``routing_scores_vmem`` is [batch_size, 1024]. Paired ranks exchange their
    512-score halves directly in that padded workspace, then every top-k
    iteration advances all batch rows together. Selected local routes are
    returned as [batch_size, top_k] values for scalar SMEM materialization by
    the caller.
    """
    rank = jax.lax.axis_index("tp")
    batch_size = hidden_states.shape[0]
    expert_count = routing_bias_ref.shape[-1]
    if expert_count != 896 or hidden_states.shape[-1] > 7168:
        raise ValueError("Host-shared router requires 896 expert scores")
    if routing_scores_vmem.shape[-1] != 1024:
        raise ValueError("Routing score workspace must have padded width 1024")

    router_lane = rank % 2
    local_scores = _dot(hidden_states, routing_projection_vmem[0]).reshape(
        batch_size, 512
    )
    score_half = routing_scores_vmem.at[
        :, pl.ds(router_lane * 512, 512)
    ]
    score_half[...] = local_scores
    copy = tpu.make_async_remote_copy(
        score_half,
        score_half,
        send_semaphore,
        receive_semaphore,
        device_id=(rank ^ 1,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    copy.start()
    copy.wait()

    raw_scores = routing_scores_vmem[:, :expert_count]
    uncorrected_probabilities = jax.nn.sigmoid(raw_scores)
    routing_scores_vmem[:, :expert_count] = (
        uncorrected_probabilities + routing_bias_ref[...]
    )
    selected_experts = jnp.zeros(
        (batch_size, selected_expert_count), jnp.int32
    )
    selected_probabilities = jnp.zeros(
        (batch_size, selected_expert_count), jnp.float32
    )
    local_route_counts = jnp.zeros((batch_size,), jnp.int32)
    probability_totals = jnp.zeros((batch_size,), jnp.float32)
    expert_ids = jnp.arange(expert_count)[None, :]
    route_slots = jnp.arange(selected_expert_count)[None, :]
    # Only top-k rank is sequential. Every score reduction and ownership
    # operation carries the leading batch dimension with no per-row control.
    for selection in range(selected_expert_count):
        winners = jnp.argmax(
            routing_scores_vmem[:, :expert_count], axis=1
        ).astype(jnp.int32)
        winner_mask = expert_ids == winners[:, None]
        probabilities = jnp.sum(
            jnp.where(winner_mask, uncorrected_probabilities, 0), axis=1
        )
        probability_totals += probabilities
        routing_scores_vmem[:, :expert_count] = jnp.where(
            winner_mask, -jnp.inf, routing_scores_vmem[:, :expert_count]
        )

        counts_before_selection = local_route_counts
        owned_locally = (
            winners // local_expert_count
            == rank // expert_tensor_parallel
        )
        destination_mask = (
            owned_locally[:, None]
            & (route_slots == counts_before_selection[:, None])
        )
        selected_experts = jnp.where(
            destination_mask,
            (winners % local_expert_count)[:, None],
            selected_experts,
        )
        selected_probabilities = jnp.where(
            destination_mask,
            probabilities[:, None],
            selected_probabilities,
        )
        local_route_counts = (
            counts_before_selection + owned_locally.astype(jnp.int32)
        )
        on_local_selection(
            winners, owned_locally, counts_before_selection, selected_experts
        )
        after_selection(selection)

    return (
        selected_experts,
        selected_probabilities,
        local_route_counts,
        probability_totals,
    )


def _start_initial_expert_weight_copies(
    selected_experts,
    local_route_counts,
    copy_expert,
    *,
    slot_count,
    already_started,
    group_pending_routes,
):
    """Prefetch the first routed experts in their execution order.

    This uses the same cross-row ordering as the expert stream, so it works for
    both shared and divergent routes without specializing on batch size.
    """
    selected_expert_count = selected_experts.shape[1]
    route_slots = jnp.arange(selected_expert_count)[None, :]
    processed = jnp.zeros(selected_experts.shape, jnp.bool_)

    def next_expert(completed):
        pending = (route_slots < local_route_counts[:, None]) & ~completed
        if group_pending_routes:
            minimum_expert = jnp.min(
                jnp.where(
                    pending,
                    selected_experts.astype(jnp.float32),
                    jnp.inf,
                )
            )
            has_expert = jnp.any(pending)
            expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
            consumed = pending & (selected_experts == expert)
        else:
            rows_with_routes = jnp.any(pending, axis=1)
            current_slots = jnp.sum(completed.astype(jnp.int32), axis=1)
            current_slot_mask = route_slots == current_slots[:, None]
            current_experts = jnp.sum(
                jnp.where(current_slot_mask, selected_experts, 0), axis=1
            )
            minimum_expert = jnp.min(
                jnp.where(
                    rows_with_routes,
                    current_experts.astype(jnp.float32),
                    jnp.inf,
                )
            )
            has_expert = jnp.any(rows_with_routes)
            expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
            consumed = (
                rows_with_routes & (current_experts == expert)
            )[:, None] & current_slot_mask
        return expert, consumed, has_expert

    for slot in range(slot_count):
        expert, consumed, has_expert = next_expert(processed)

        @pl.when(has_expert & (slot >= already_started))
        def start_copy(expert=expert, slot=slot):
            _start_async_copies(copy_expert(expert, slot))

        processed |= consumed
def _quantized_expert_dot(
    source,
    packed,
    scales,
    accumulator,
    contraction_size,
    output_size,
    *,
    block_major=False,
):
    """Multiply by one resident MXFP4 matrix or reblocked FP8 matrix.

    Reblocked FP8 stores one BF16 power-of-two scale for each contiguous
    contraction block. The scale-row count determines the static block width.
    This trades a larger HBM representation for removing MXFP4 unpack and its
    per-32-element scaled partial reduction.
    """
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)

    if packed.dtype == jnp.float8_e4m3fn:
        scale_block_count = scales.shape[0]
        if contraction_size % scale_block_count:
            raise ValueError("FP8 scale blocks must divide the contraction")
        scale_block_size = contraction_size // scale_block_count
        if scale_block_size % 128:
            raise ValueError("FP8 scale blocks must be MXU aligned")

        # Pallas traces and unrolls this fixed-size loop. Each FP8 tile is
        # consumed directly by MXU and scaled only after the dot product.
        for scale_block in range(scale_block_count):
            start = scale_block * scale_block_size
            activation = source[
                :, pl.ds(pl.multiple_of(start, 128), scale_block_size)
            ].astype(jnp.bfloat16)
            values = packed[start : start + scale_block_size, :]
            partial = jax.lax.dot_general(
                jnp.pad(
                    activation,
                    ((0, 16 - activation.shape[0]), (0, 0)),
                ),
                values,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )[: activation.shape[0]]
            accumulator[...] += (
                partial * scales[scale_block : scale_block + 1].astype(jnp.float32)
            )
        return

    if packed.dtype != jnp.uint32:
        raise ValueError("Expert weights must use packed MXFP4 or reblocked FP8")

    split_small_contraction = contraction_size == 768 and block_major
    block_size = 256 if split_small_contraction else min(2048, contraction_size)

    def tile(start, size):
        values = tpu.bitcast(
            packed[
                pl.ds(pl.multiple_of(start // 8, 8), size // 8), :
            ],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        if split_small_contraction:
            # The full 24-row view is aligned even though the second static
            # 512-wide tile begins halfway through a 32-row VMEM tile.
            exponents = scales[...][
                start // 32 : (start + size) // 32, :
            ].astype(jnp.uint32)
        else:
            exponents = scales[
                pl.ds(pl.multiple_of(start // 32, 32), size // 32), :
            ].astype(jnp.uint32)
        scale = jax.lax.bitcast_convert_type(exponents << 23, jnp.float32)
        activation_alignment = 128 if split_small_contraction else 512
        activation = source[
            :, pl.ds(pl.multiple_of(start, activation_alignment), size)
        ].astype(jnp.bfloat16)
        scale_block_mask = (
            jnp.arange(size // 32)[:, None]
            == jnp.arange(size)[None, :] // 32
        )
        if split_small_contraction:
            left = jnp.where(
                scale_block_mask[:, None, :],
                activation[None, :, :],
                jnp.bfloat16(0),
            )
            partial = jax.lax.dot_general(
                left,
                values,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            accumulator[...] += jnp.sum(
                partial * scale[:, None, :], axis=0
            )
        else:
            left = jnp.where(
                scale_block_mask[None], activation[:, None, :], jnp.bfloat16(0)
            )
            partial = jax.lax.dot_general(
                left,
                values,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            accumulator[...] += jnp.sum(partial * scale[None], axis=1)

    if split_small_contraction:
        for start in range(0, contraction_size, block_size):
            tile(start, block_size)
    else:
        @pl.loop(0, contraction_size // block_size)
        def full_tile(index):
            tile(index * block_size, block_size)

        if contraction_size % block_size:
            tile(
                contraction_size // block_size * block_size,
                contraction_size % block_size,
            )


def _paired_expert_gate_up(
    source,
    gate_up_vmem,
    gate_up_scale_vmem,
    accumulator,
    first_slot,
    second_slot,
    contraction_size,
    *,
    second_source=None,
):
    """Run two resident expert gate/up projections for every batch row."""
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)

    if gate_up_vmem.dtype == jnp.float8_e4m3fn:
        scale_block_count = gate_up_scale_vmem.shape[1]
        if contraction_size % scale_block_count:
            raise ValueError("FP8 gate/up scales must divide the contraction")
        scale_block_size = contraction_size // scale_block_count
        if scale_block_size % 128:
            raise ValueError("FP8 gate/up scale blocks must be MXU aligned")
        for scale_block in range(scale_block_count):
            start = scale_block * scale_block_size
            values = jnp.concatenate(
                (
                    gate_up_vmem[
                        first_slot,
                        pl.ds(pl.multiple_of(start, 128), scale_block_size),
                        :,
                    ],
                    gate_up_vmem[
                        second_slot,
                        pl.ds(pl.multiple_of(start, 128), scale_block_size),
                        :,
                    ],
                ),
                axis=1,
            )
            scales = jnp.concatenate(
                (
                    gate_up_scale_vmem[
                        first_slot, scale_block : scale_block + 1, :
                    ],
                    gate_up_scale_vmem[
                        second_slot, scale_block : scale_block + 1, :
                    ],
                ),
                axis=1,
            ).astype(jnp.float32)
            activation = source[
                :, pl.ds(pl.multiple_of(start, 128), scale_block_size)
            ].astype(jnp.bfloat16)
            if second_source is not None:
                activation = jnp.concatenate(
                    (
                        activation,
                        second_source[
                            :,
                            pl.ds(
                                pl.multiple_of(start, 128), scale_block_size
                            ),
                        ].astype(jnp.bfloat16),
                    ),
                    axis=0,
                )
            partial = jax.lax.dot_general(
                jnp.pad(
                    activation,
                    ((0, 16 - activation.shape[0]), (0, 0)),
                ),
                values,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )[: activation.shape[0]]
            accumulator[...] += partial * scales
        return

    def tile(start, size):
        packed = jnp.concatenate(
            (
                gate_up_vmem[
                    first_slot, pl.ds(start // 8, size // 8), :
                ],
                gate_up_vmem[
                    second_slot, pl.ds(start // 8, size // 8), :
                ],
            ),
            axis=1,
        )
        values = tpu.bitcast(packed, jnp.float4_e2m1fn).astype(
            jnp.float8_e4m3fn
        )
        exponents = jnp.concatenate(
            (
                gate_up_scale_vmem[
                    first_slot, pl.ds(start // 32, size // 32), :
                ],
                gate_up_scale_vmem[
                    second_slot, pl.ds(start // 32, size // 32), :
                ],
            ),
            axis=1,
        ).astype(jnp.uint32)
        scale = jax.lax.bitcast_convert_type(exponents << 23, jnp.float32)
        activation = source[:, pl.ds(start, size)].astype(jnp.bfloat16)
        if second_source is not None:
            activation = jnp.concatenate(
                (
                    activation,
                    second_source[:, pl.ds(start, size)].astype(jnp.bfloat16),
                ),
                axis=0,
            )
        scale_block_mask = (
            jnp.arange(size // 32)[:, None]
            == jnp.arange(size)[None, :] // 32
        )
        left = jnp.where(
            scale_block_mask[None], activation[:, None, :], jnp.bfloat16(0)
        )
        partial = jax.lax.dot_general(
            left,
            values,
            (((2,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        accumulator[...] += jnp.sum(partial * scale[None], axis=1)

    # Narrow scaled partials shorten register live ranges; 512 is the measured
    # B8 optimum on TPU v6e.
    tile_size = 512
    for start in range(0, contraction_size, tile_size):
        tile(start, min(tile_size, contraction_size - start))


def _run_routed_expert_stream(
    source,
    selected_experts,
    selected_probabilities,
    local_route_counts,
    probability_totals,
    expert_gate_up_vmem,
    expert_gate_up_scale_vmem,
    expert_down_vmem,
    expert_down_scale_vmem,
    expert_output,
    gate_up_accumulator,
    down_accumulator,
    processed_routes_vmem,
    first_compact_source_vmem,
    second_compact_source_vmem,
    copy_expert,
    *,
    expert_input_size,
    expert_width,
    expert_output_size,
    group_pending_routes,
):
    """Execute the next route shared by any ready batch rows.

    Rows retain their own routing order. When several rows have the same next
    expert, its weights are loaded once and its projections carry the batch
    dimension. The loop is bounded by route count, never by batch size.
    """
    batch_size, selected_expert_count = selected_experts.shape
    route_slots = jnp.arange(selected_expert_count)[None, :]
    processed_routes_vmem[...] = jnp.zeros(
        processed_routes_vmem.shape, jnp.int32
    )
    expert_output[...] = jnp.zeros(expert_output.shape, jnp.float32)

    def next_expert(processed):
        pending = (
            route_slots < local_route_counts[:, None]
        ) & ~processed
        if group_pending_routes:
            minimum_expert = jnp.min(
                jnp.where(
                    pending,
                    selected_experts.astype(jnp.float32),
                    jnp.inf,
                )
            )
            has_expert = jnp.any(pending)
            expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
            selected_slot_mask = pending & (selected_experts == expert)
            active_rows = jnp.any(selected_slot_mask, axis=1)
            return (
                expert,
                active_rows,
                selected_slot_mask,
                selected_slot_mask,
                has_expert,
            )
        rows_with_routes = jnp.any(pending, axis=1)
        current_slots = jnp.sum(processed.astype(jnp.int32), axis=1)
        current_slot_mask = route_slots == current_slots[:, None]
        current_experts = jnp.sum(
            jnp.where(current_slot_mask, selected_experts, 0), axis=1
        )
        minimum_expert = jnp.min(
            jnp.where(
                rows_with_routes,
                current_experts.astype(jnp.float32),
                jnp.inf,
            )
        )
        has_expert = jnp.any(rows_with_routes)
        expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
        active_rows = rows_with_routes & (current_experts == expert)
        consumed = active_rows[:, None] & current_slot_mask
        return expert, active_rows, current_slot_mask, consumed, has_expert

    def finish_expert(active_rows, current_slot_mask, copies, column, slot):
        gate_up = gate_up_accumulator[
            :, column : column + 2 * expert_width
        ].astype(jnp.bfloat16).astype(jnp.float32)
        middle = _gated_activation(
            gate_up[:, :expert_width], gate_up[:, expert_width:]
        )
        gate_up_accumulator[:, :expert_width] = middle.astype(jnp.float32)
        copies.down.wait()
        copies.down_scales.wait()
        _quantized_expert_dot(
            gate_up_accumulator.at[:, :expert_width],
            expert_down_vmem.at[slot],
            expert_down_scale_vmem.at[slot],
            down_accumulator,
            expert_width,
            expert_output_size,
            # The scale-block-major layout pays off for a full B8 tile but
            # regresses the compact singleton path below.
            block_major=batch_size >= 8,
        )
        route_probabilities = jnp.sum(
            jnp.where(
                current_slot_mask,
                selected_probabilities,
                jnp.float32(0),
            ),
            axis=1,
        ) / probability_totals
        expert_output[:, :expert_output_size] += (
            down_accumulator[...]
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
            * jnp.where(
                active_rows, route_probabilities, jnp.float32(0)
            )[:, None]
        )

    def finish_singleton_expert(
        active_rows,
        current_slot_mask,
        copies,
        compact_row,
        column,
        slot,
    ):
        """Finish one expert whose route is active for exactly one batch row."""
        gate_up = gate_up_accumulator[
            compact_row : compact_row + 1,
            column : column + 2 * expert_width,
        ].astype(jnp.bfloat16).astype(jnp.float32)
        middle = _gated_activation(
            gate_up[:, :expert_width], gate_up[:, expert_width:]
        )
        gate_up_accumulator[
            compact_row : compact_row + 1, :expert_width
        ] = middle.astype(jnp.float32)
        copies.down.wait()
        copies.down_scales.wait()
        _quantized_expert_dot(
            gate_up_accumulator.at[
                compact_row : compact_row + 1, :expert_width
            ],
            expert_down_vmem.at[slot],
            expert_down_scale_vmem.at[slot],
            down_accumulator.at[:1],
            expert_width,
            expert_output_size,
        )
        route_probabilities = jnp.sum(
            jnp.where(
                current_slot_mask,
                selected_probabilities,
                jnp.float32(0),
            ),
            axis=1,
        ) / probability_totals
        singleton_output = (
            down_accumulator[:1]
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        expert_output[:, :expert_output_size] += jnp.where(
            active_rows[:, None],
            singleton_output * route_probabilities[:, None],
            jnp.float32(0),
        )

    def stage_singleton_source(active_rows, destination):
        active_row_bits = active_rows.astype(jnp.int32)
        for row in range(batch_size):
            @pl.when(active_row_bits[row] != 0)
            def copy_active_row():
                destination[...] = source[
                    row : row + 1, :expert_input_size
                ]

    valid_routes = route_slots < local_route_counts[:, None]
    routes_are_shared = (not group_pending_routes) & jnp.all(
        local_route_counts == local_route_counts[0]
    ) & jnp.all(
        ~valid_routes
        | (selected_experts == selected_experts[0:1])
    )

    @pl.when(routes_are_shared)
    def run_shared_routes():
        def shared_expert_at(index):
            return jnp.sum(
                jnp.where(
                    route_slots[0] == index,
                    selected_experts[0],
                    jnp.int32(0),
                )
            ).astype(jnp.int32)

        def shared_probabilities_at(index):
            return jnp.sum(
                jnp.where(
                    route_slots == index,
                    selected_probabilities,
                    jnp.float32(0),
                ),
                axis=1,
            )

        def wait_for_gate_up(index, slot):
            copies = copy_expert(
                shared_expert_at(index), slot, down=False
            )
            copies.gate_up.wait()
            copies.gate_up_scales.wait()

        def wait_for_down(index, slot):
            copies = copy_expert(
                shared_expert_at(index), slot, gate_up=False
            )
            copies.down.wait()
            copies.down_scales.wait()

        def finish_shared_expert(index, slot, column):
            gate_up = gate_up_accumulator[
                :, column : column + 2 * expert_width
            ].astype(jnp.bfloat16).astype(jnp.float32)
            middle = _gated_activation(
                gate_up[:, :expert_width], gate_up[:, expert_width:]
            )
            gate_up_accumulator[:, :expert_width] = middle.astype(
                jnp.float32
            )
            wait_for_down(index, slot)
            _quantized_expert_dot(
                gate_up_accumulator.at[:, :expert_width],
                expert_down_vmem.at[slot],
                expert_down_scale_vmem.at[slot],
                down_accumulator,
                expert_width,
                expert_output_size,
                block_major=batch_size >= 8,
            )
            route_weights = (
                shared_probabilities_at(index) / probability_totals
            )[:, None]
            expert_output[:, :expert_output_size] += (
                down_accumulator[...]
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
                * route_weights
            )

            @pl.when(index + 4 < local_route_counts[0])
            def refill_expert():
                _start_async_copies(
                    copy_expert(shared_expert_at(index + 4), slot)
                )

        @pl.loop(0, local_route_counts[0] // 2)
        def run_shared_pair(pair):
            first = 2 * pair
            first_slot = first % 4
            second_slot = (first + 1) % 4
            wait_for_gate_up(first, first_slot)
            wait_for_gate_up(first + 1, second_slot)
            _paired_expert_gate_up(
                source,
                expert_gate_up_vmem,
                expert_gate_up_scale_vmem,
                gate_up_accumulator,
                first_slot,
                second_slot,
                expert_input_size,
            )
            finish_shared_expert(first, first_slot, 0)
            finish_shared_expert(
                first + 1, second_slot, 2 * expert_width
            )

        @pl.when(local_route_counts[0] % 2 == 1)
        def run_shared_tail():
            index = local_route_counts[0] - 1
            slot = index % 4
            wait_for_gate_up(index, slot)
            _quantized_expert_dot(
                source,
                expert_gate_up_vmem.at[slot],
                expert_gate_up_scale_vmem.at[slot],
                gate_up_accumulator.at[:, : 2 * expert_width],
                expert_input_size,
                2 * expert_width,
            )
            finish_shared_expert(index, slot, 0)

    @pl.when(~routes_are_shared)
    def run_divergent_routes():
        @pl.loop(0, (jnp.sum(local_route_counts) + 1) // 2)
        def run_next_expert_pair(iteration):
            processed = processed_routes_vmem[:, :selected_expert_count] != 0
            (
                first_expert,
                first_rows,
                first_slot_mask,
                first_consumed,
                has_first,
            ) = next_expert(processed)
            (
                second_expert,
                second_rows,
                second_slot_mask,
                second_consumed,
                has_second,
            ) = next_expert(processed | first_consumed)
            processed_after_pair = processed | first_consumed | second_consumed
            (
                third_expert,
                _,
                _,
                third_consumed,
                has_third,
            ) = next_expert(processed_after_pair)
            (
                fourth_expert,
                _,
                _,
                _,
                has_fourth,
            ) = next_expert(processed_after_pair | third_consumed)

            current_base_slot = (iteration % 2) * 2
            next_base_slot = ((iteration + 1) % 2) * 2
            first_copies = copy_expert(first_expert, current_base_slot)
            second_copies = copy_expert(second_expert, current_base_slot + 1)
            third_copies = copy_expert(third_expert, next_base_slot)
            fourth_copies = copy_expert(fourth_expert, next_base_slot + 1)

            should_prefetch_third = has_third & (iteration != 0)
            should_prefetch_fourth = has_fourth & (iteration != 0)

            @pl.when(should_prefetch_third)
            def prefetch_third():
                _start_async_copies(third_copies)

            @pl.when(should_prefetch_fourth)
            def prefetch_fourth():
                _start_async_copies(fourth_copies)

            def execute_full_pair():
                first_copies.gate_up.wait()
                first_copies.gate_up_scales.wait()
                second_copies.gate_up.wait()
                second_copies.gate_up_scales.wait()
                _paired_expert_gate_up(
                    source,
                    expert_gate_up_vmem,
                    expert_gate_up_scale_vmem,
                    gate_up_accumulator,
                    current_base_slot,
                    current_base_slot + 1,
                    expert_input_size,
                )
                finish_expert(
                    first_rows,
                    first_slot_mask,
                    first_copies,
                    0,
                    current_base_slot,
                )
                finish_expert(
                    second_rows,
                    second_slot_mask,
                    second_copies,
                    2 * expert_width,
                    current_base_slot + 1,
                )

            def execute_full_tail():
                first_copies.gate_up.wait()
                first_copies.gate_up_scales.wait()
                _quantized_expert_dot(
                    source,
                    expert_gate_up_vmem.at[current_base_slot],
                    expert_gate_up_scale_vmem.at[current_base_slot],
                    gate_up_accumulator.at[:, : 2 * expert_width],
                    expert_input_size,
                    2 * expert_width,
                )
                finish_expert(
                    first_rows,
                    first_slot_mask,
                    first_copies,
                    0,
                    current_base_slot,
                )

            # At B8, running a full padded MXU tile for a singleton route is
            # more expensive than staging its one active row. At B2/B4 the
            # selection overhead is larger than the saved padded work.
            if batch_size >= 8:
                first_row_count = jnp.sum(first_rows.astype(jnp.int32))
                second_row_count = jnp.sum(second_rows.astype(jnp.int32))
                compact_pair = (
                    has_first
                    & has_second
                    & (first_row_count == 1)
                    & (second_row_count == 1)
                )

                @pl.when(compact_pair)
                def execute_compact_pair():
                    first_copies.gate_up.wait()
                    first_copies.gate_up_scales.wait()
                    second_copies.gate_up.wait()
                    second_copies.gate_up_scales.wait()
                    stage_singleton_source(
                        first_rows, first_compact_source_vmem
                    )
                    stage_singleton_source(
                        second_rows, second_compact_source_vmem
                    )
                    _paired_expert_gate_up(
                        first_compact_source_vmem,
                        expert_gate_up_vmem,
                        expert_gate_up_scale_vmem,
                        gate_up_accumulator.at[:2],
                        current_base_slot,
                        current_base_slot + 1,
                        expert_input_size,
                        second_source=second_compact_source_vmem,
                    )
                    finish_singleton_expert(
                        first_rows,
                        first_slot_mask,
                        first_copies,
                        0,
                        0,
                        current_base_slot,
                    )
                    finish_singleton_expert(
                        second_rows,
                        second_slot_mask,
                        second_copies,
                        1,
                        2 * expert_width,
                        current_base_slot + 1,
                    )

                @pl.when(has_first & has_second & ~compact_pair)
                def execute_noncompact_pair():
                    execute_full_pair()

                compact_tail = (
                    has_first & ~has_second & (first_row_count == 1)
                )

                @pl.when(compact_tail)
                def execute_compact_tail():
                    first_copies.gate_up.wait()
                    first_copies.gate_up_scales.wait()
                    stage_singleton_source(
                        first_rows, first_compact_source_vmem
                    )
                    _quantized_expert_dot(
                        first_compact_source_vmem,
                        expert_gate_up_vmem.at[current_base_slot],
                        expert_gate_up_scale_vmem.at[current_base_slot],
                        gate_up_accumulator.at[:1, : 2 * expert_width],
                        expert_input_size,
                        2 * expert_width,
                    )
                    finish_singleton_expert(
                        first_rows,
                        first_slot_mask,
                        first_copies,
                        0,
                        0,
                        current_base_slot,
                    )

                @pl.when(
                    has_first
                    & ~has_second
                    & (first_row_count != 1)
                )
                def execute_noncompact_tail():
                    execute_full_tail()
            else:
                @pl.when(has_first & has_second)
                def execute_pair():
                    execute_full_pair()

                @pl.when(has_first & ~has_second)
                def execute_tail():
                    execute_full_tail()

            processed_routes_vmem[:, :selected_expert_count] = jnp.where(
                first_consumed | second_consumed,
                jnp.int32(1),
                processed_routes_vmem[:, :selected_expert_count],
            )


def communication_scratch(batch_size: int = 1) -> tuple[Any, ...]:
    """Shared for the entire fused stack, including the reduction sequence."""
    # Shared outputs and attention projections use 28 FP32 transport rows per
    # token. Routed-expert outputs are half as wide. Keep those roles separate
    # so the narrower packet and fixed native-row attention backing do not
    # inherit the widest batch-scaled allocation.
    wide_reduction_rows = max(64, 28 * batch_size)
    # B6 uses the eight-lane routed reduce-scatter with two zero-padded token
    # shards, so it needs the same transport capacity as B8.
    expert_reduction_rows = max(32, 14 * batch_size, 112 if batch_size == 6 else 0)
    return (
        tpu.VMEM((8, wide_reduction_rows, 128), jnp.float32),
        tpu.VMEM((8, expert_reduction_rows, 128), jnp.float32),
        tpu.VMEM((4, wide_reduction_rows, 128), jnp.float32),
        tpu.VMEM((4, expert_reduction_rows, 128), jnp.float32),
        # Native MXU row backing used by attention's group gather. The active
        # batch view is supplied by _bf16_rows_view.
        tpu.VMEM((16, 1024), jnp.bfloat16),
        tpu.VMEM((16, 7168), jnp.bfloat16),
        tpu.SemaphoreType.DMA((16,)),
        tpu.SemaphoreType.DMA((16,)),
        tpu.SMEM((1,), jnp.int32),
        tpu.VMEM((batch_size, 4096), jnp.bfloat16),
    )


TensorLike = jax_typing.ArrayLike | jax.ShapeDtypeStruct


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class AttentionCommunicationWorkspace:
    """Collective transport storage shared by attention and MoE layers."""

    shared_local_reduction_vmem: Any  # [8, max(64, 28 * batch_size), 128]
    expert_local_reduction_vmem: Any  # [8, max(32, 14 * batch_size), 128]
    shared_host_reduction_vmem: Any  # [4, max(64, 28 * batch_size), 128]
    expert_host_reduction_vmem: Any  # [4, max(32, 14 * batch_size), 128]
    attention_features_vmem: Any  # [16, 1024]
    attention_output_vmem: Any  # [16, 7168]
    send_semaphores: Any  # [16]
    receive_semaphores: Any  # [16]
    reduction_phase: Any  # [1]
    latent_gather_vmem: Any  # [batch_size, 4096]


def _start_async_copies(copies: SimpleNamespace) -> None:
    for copy in vars(copies).values():
        if copy is not None:
            copy.start()


def _gather_output_shards(
    output_vmem,
    send_semaphores,
    receive_semaphores,
    semaphore_slot,
) -> None:
    rank = jax.lax.axis_index("tp")
    local_shard = output_vmem.at[:, pl.ds((rank % 8) * 1024, 1024)]
    for offset in (7, 6, 5, 3, 4, 2, 1):
        tpu.make_async_remote_copy(
            local_shard,
            local_shard,
            send_semaphores.at[semaphore_slot],
            receive_semaphores.at[semaphore_slot],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
    received_shards = output_vmem.at[:, : 7 * 1024]
    tpu.make_async_copy(
        received_shards,
        received_shards,
        send_semaphores.at[semaphore_slot],
    ).wait()
    tpu.make_async_copy(
        received_shards,
        received_shards,
        receive_semaphores.at[semaphore_slot],
    ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class MixtureOfExpertsWeights:
    """MoE parameters and persistent resources for asynchronous weight copies."""

    routing_projection: TensorLike  # [hidden_size, 896]
    routing_projection_vmem: Any  # [1, hidden_size, 512]
    routing_copy_semaphores: Any  # [2]
    routing_bias: TensorLike  # [1, 896]
    latent_input_projection: TensorLike  # [hidden_size, 128]
    latent_output_projection: TensorLike  # [expert_output_size, 1024]
    latent_output_projection_vmem: Any  # [expert_output_size, 1024]
    latent_output_copy_semaphore: Any  # scalar DMA semaphore
    latent_normalization: TensorLike  # [1, padded_latent_size]
    shared_gate_up_projection: TensorLike  # [hidden_size, 2 * shared_width]
    shared_down_projection: TensorLike  # [shared_width, hidden_size]
    # Packed MXFP4: [local_experts, expert_input_size / 8, 2 * expert_width].
    # Reblocked FP8: [local_experts, expert_input_size, 2 * expert_width].
    expert_gate_up_projection: TensorLike
    expert_gate_up_projection_vmem: Any  # [4, ...]
    expert_gate_up_scales: TensorLike
    expert_gate_up_scales_vmem: Any  # [4, ...]
    # Packed MXFP4: [local_experts, expert_width / 8, expert_output_size].
    # Reblocked FP8: [local_experts, expert_width, expert_output_size].
    expert_down_projection: TensorLike
    expert_down_projection_vmem: Any  # [4, ...]
    expert_down_scales: TensorLike
    expert_down_scales_vmem: Any  # [4, ...]
    expert_copy_semaphores: Any  # [4, 4]

    def _copy_routing_projection(
        self,
        *,
        wait: bool,
        initialize_padding: bool = True,
    ) -> None:
        lane = jax.lax.axis_index("tp") % 2

        @pl.when(lane == 0)
        def copy_full_half():
            copy = tpu.make_async_copy(
                self.routing_projection.at[:, :512],
                self.routing_projection_vmem.at[0],
                self.routing_copy_semaphores.at[0],
            )
            copy.wait() if wait else copy.start()

        @pl.when(lane == 1)
        def copy_tail():
            if not wait and initialize_padding:
                self.routing_projection_vmem[0, :, 384:] = jnp.zeros(
                    (self.routing_projection.shape[0], 128),
                    jnp.bfloat16,
                )
            copy = tpu.make_async_copy(
                self.routing_projection.at[:, pl.ds(512, 384)],
                self.routing_projection_vmem.at[0, :, :384],
                self.routing_copy_semaphores.at[0],
            )
            copy.wait() if wait else copy.start()

    def start_routing_projection_copy(
        self, *, initialize_padding: bool = True
    ) -> None:
        self._copy_routing_projection(
            wait=False,
            initialize_padding=initialize_padding,
        )

    def wait_for_routing_projection_copy(self) -> None:
        self._copy_routing_projection(wait=True)

    def make_latent_output_copy(self):
        return tpu.make_async_copy(
            self.latent_output_projection,
            self.latent_output_projection_vmem,
            self.latent_output_copy_semaphore,
        )

    def make_expert_copies(
        self,
        expert,
        slot,
        *,
        gate_up: bool = True,
        down: bool = True,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            gate_up=(
                tpu.make_async_copy(
                    self.expert_gate_up_projection.at[expert],
                    self.expert_gate_up_projection_vmem.at[slot],
                    self.expert_copy_semaphores.at[slot, 0],
                )
                if gate_up
                else None
            ),
            gate_up_scales=(
                tpu.make_async_copy(
                    self.expert_gate_up_scales.at[expert],
                    self.expert_gate_up_scales_vmem.at[slot],
                    self.expert_copy_semaphores.at[slot, 1],
                )
                if gate_up
                else None
            ),
            down=(
                tpu.make_async_copy(
                    self.expert_down_projection.at[expert],
                    self.expert_down_projection_vmem.at[slot],
                    self.expert_copy_semaphores.at[slot, 2],
                )
                if down
                else None
            ),
            down_scales=(
                tpu.make_async_copy(
                    self.expert_down_scales.at[expert],
                    self.expert_down_scales_vmem.at[slot],
                    self.expert_copy_semaphores.at[slot, 3],
                )
                if down
                else None
            ),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class MixtureOfExpertsWorkspace:
    """Persistent VMEM and semaphores used by one MoE invocation."""

    selected_experts_vmem: Any  # [batch_size, 128]
    selected_probabilities_vmem: Any  # [batch_size, 128]
    local_route_counts_vmem: Any  # [batch_size, 128]
    probability_totals_vmem: Any  # [batch_size, 128]
    routing_scores_vmem: Any  # [batch_size, 1024]
    shared_outputs_vmem: Any  # [batch_size, hidden_size]
    routed_outputs_vmem: Any  # [batch_size, padded_latent_size]
    gate_up_accumulator_vmem: Any  # [batch_size, 4 * expert_width]
    down_accumulator_vmem: Any  # [batch_size, expert_output_size]
    processed_routes_vmem: Any  # [batch_size, 128]
    first_compact_source_vmem: Any  # [1, expert_input_size]
    second_compact_source_vmem: Any  # [1, expert_input_size]
    shared_send_semaphores: Any  # [10]
    shared_receive_semaphores: Any  # [10]
    output_vmem: Any  # [batch_size, 8192] or token-scatter transport shape
    output_send_semaphores: Any  # [2] or [7]
    output_receive_semaphores: Any  # [2] or [7]
    latent_send_semaphores: Any  # [2]
    latent_receive_semaphores: Any  # [2]


def moe(
    hidden_ref: TensorLike,  # [batch_size, hidden_size]
    output_ref: TensorLike,  # [batch_size, hidden_size] or owned token rows
    *,
    weights: MixtureOfExpertsWeights,
    workspace: MixtureOfExpertsWorkspace,
    communication_workspace: AttentionCommunicationWorkspace,
    selected_expert_count: int = 16,
    epsilon: float = 1e-5,
    latent_feature_count: int = 3584,
    synchronize_devices: bool = True,
    routing_projection_is_prefetched: bool = False,
    group_pending_expert_routes: bool = False,
    token_scatter_output: bool = False,
    routed_reduce_scatter: bool = False,
    after_experts=None,
) -> None:
    """Execute one mixture-of-experts layer inside a Pallas program."""
    batch_size, hidden_size = hidden_ref.shape
    expert_count = weights.routing_projection.shape[1]
    latent_size = weights.latent_normalization.shape[-1]
    expert_output_size = weights.expert_down_projection.shape[-1]
    local_expert_count = weights.expert_gate_up_projection.shape[0]
    expert_tensor_parallel = 32 * local_expert_count // expert_count
    down_is_reblocked_fp8 = (
        weights.expert_down_projection.dtype == jnp.float8_e4m3fn
    )
    expert_width = (
        weights.expert_down_projection.shape[1]
        if down_is_reblocked_fp8
        else weights.expert_down_projection.shape[1] * 8
    )
    expert_input_size = (
        weights.expert_gate_up_projection.shape[1] * 8
        if weights.expert_gate_up_projection.dtype == jnp.uint32
        else weights.expert_gate_up_projection.shape[1]
    )
    shared_width = weights.shared_down_projection.shape[0]
    expert_prefetch_slot_count = 4

    if not 1 <= batch_size <= 16:
        raise ValueError("MoE supports native MXU tiles from 1 to 16 rows")
    if token_scatter_output and batch_size != 8:
        raise ValueError("MoE token-scattered output requires batch size 8")
    if routed_reduce_scatter and batch_size not in (6, 8):
        raise ValueError("MoE routed reduce-scatter requires batch size 6 or 8")
    if weights.expert_gate_up_projection.dtype not in (
        jnp.uint32,
        jnp.float8_e4m3fn,
    ) or weights.expert_down_projection.dtype not in (
        jnp.uint32,
        jnp.float8_e4m3fn,
    ):
        raise ValueError("Unsupported routed-expert weight storage")
    if expert_tensor_parallel not in (1, 2, 4):
        raise ValueError("Expert tensor parallelism must be 1, 2, or 4")
    if (
        hidden_size % 128
        or expert_count % 128
        or latent_size % 256
        or expert_width % 256
        or hidden_size > 8192
        or latent_size > 8192
    ):
        raise ValueError("Unsupported MoE tile dimensions")
    if (
        weights.latent_input_projection.shape != (hidden_size, 128)
        or weights.latent_output_projection.shape
        != (expert_output_size, 1024)
    ):
        raise ValueError("Latent projections require padded TP32 shards")
    if (
        expert_count * expert_tensor_parallel != 32 * local_expert_count
        or expert_output_size > latent_size
        or expert_input_size > latent_size
    ):
        raise ValueError("Expert ownership or latent padding is inconsistent")
    if expert_input_size % 512 or expert_output_size % 128:
        raise ValueError("Expert dimensions must satisfy MXFP4 alignment")

    if weights.expert_gate_up_projection.dtype == jnp.uint32:
        if (
            weights.expert_gate_up_projection.shape[1:]
            != (expert_input_size // 8, 2 * expert_width)
            or weights.expert_gate_up_scales.shape[1:]
            != (expert_input_size // 32, 2 * expert_width)
            or weights.expert_gate_up_scales.dtype != jnp.uint8
        ):
            raise ValueError("MXFP4 gate/up weights have inconsistent shapes")
    elif (
        weights.expert_gate_up_projection.shape[1:]
        != (expert_input_size, 2 * expert_width)
        or weights.expert_gate_up_scales.dtype != jnp.bfloat16
        or not weights.expert_gate_up_scales.shape[1]
        or expert_input_size % weights.expert_gate_up_scales.shape[1]
    ):
        raise ValueError("FP8 gate/up weights have inconsistent shapes")

    if down_is_reblocked_fp8:
        if weights.expert_down_projection.shape[1] != expert_width:
            raise ValueError("Reblocked FP8 down weights have the wrong width")
        if weights.expert_down_scales.dtype != jnp.bfloat16:
            raise ValueError("Reblocked FP8 down scales must be BF16")
        if (
            not weights.expert_down_scales.shape[1]
            or expert_width % weights.expert_down_scales.shape[1]
        ):
            raise ValueError("FP8 down scales must partition the contraction")
    elif (
        weights.expert_down_projection.shape[1] * 8 != expert_width
        or weights.expert_down_scales.dtype != jnp.uint8
    ):
        raise ValueError("MXFP4 down weights or scales have inconsistent shapes")

    expected_output_shape = (
        (batch_size // 4, hidden_size)
        if token_scatter_output
        else hidden_ref.shape
    )
    if output_ref.shape != expected_output_shape:
        raise ValueError("MoE output has the wrong shape for its communication mode")

    expected_workspace_shapes = (
        (workspace.selected_experts_vmem.shape, (batch_size, 128)),
        (workspace.selected_probabilities_vmem.shape, (batch_size, 128)),
        (workspace.local_route_counts_vmem.shape, (batch_size, 128)),
        (workspace.probability_totals_vmem.shape, (batch_size, 128)),
        (workspace.routing_scores_vmem.shape, (batch_size, 1024)),
        (workspace.shared_outputs_vmem.shape, (batch_size, hidden_size)),
        (workspace.routed_outputs_vmem.shape, (batch_size, latent_size)),
        (
            workspace.gate_up_accumulator_vmem.shape,
            (batch_size, 4 * expert_width),
        ),
        (
            workspace.down_accumulator_vmem.shape,
            (batch_size, expert_output_size),
        ),
        (workspace.processed_routes_vmem.shape, (batch_size, 128)),
        (
            workspace.first_compact_source_vmem.shape,
            (1, expert_input_size),
        ),
        (
            workspace.second_compact_source_vmem.shape,
            (1, expert_input_size),
        ),
    )
    if any(actual != expected for actual, expected in expected_workspace_shapes):
        raise ValueError("MoE workspace has an invalid shape")

    expected_output_vmem_shape = (
        (12, batch_size // 4 * 1024 // 128, 128)
        if token_scatter_output
        else (batch_size, 8192)
    )
    if workspace.output_vmem.shape != expected_output_vmem_shape:
        raise ValueError("MoE output communication workspace has the wrong shape")
    if communication_workspace.latent_gather_vmem.shape != (batch_size, 4096):
        raise ValueError("MoE latent-gather workspace must match the batch size")
    if weights.routing_projection_vmem.shape != (1, hidden_size, 512):
        raise ValueError("MoE routing projection workspace has the wrong shape")
    expert_workspace_shapes = (
        (
            weights.expert_gate_up_projection_vmem.shape,
            (expert_prefetch_slot_count, *weights.expert_gate_up_projection.shape[1:]),
        ),
        (
            weights.expert_gate_up_scales_vmem.shape,
            (expert_prefetch_slot_count, *weights.expert_gate_up_scales.shape[1:]),
        ),
        (
            weights.expert_down_projection_vmem.shape,
            (expert_prefetch_slot_count, *weights.expert_down_projection.shape[1:]),
        ),
        (
            weights.expert_down_scales_vmem.shape,
            (expert_prefetch_slot_count, *weights.expert_down_scales.shape[1:]),
        ),
    )
    if any(actual != expected for actual, expected in expert_workspace_shapes):
        raise ValueError("MoE expert weight workspace has an invalid shape")

    expert_gate_up_vmem = weights.expert_gate_up_projection_vmem
    expert_gate_up_scale_vmem = weights.expert_gate_up_scales_vmem
    expert_down_vmem = weights.expert_down_projection_vmem
    expert_down_scale_vmem = weights.expert_down_scales_vmem
    selected_experts_vmem = workspace.selected_experts_vmem
    selected_probabilities_vmem = workspace.selected_probabilities_vmem
    local_route_counts_vmem = workspace.local_route_counts_vmem
    probability_totals_vmem = workspace.probability_totals_vmem
    shared_outputs_vmem = workspace.shared_outputs_vmem
    routed_outputs_vmem = workspace.routed_outputs_vmem
    gate_up_accumulator_vmem = workspace.gate_up_accumulator_vmem
    down_accumulator_vmem = workspace.down_accumulator_vmem
    processed_routes_vmem = workspace.processed_routes_vmem
    first_compact_source_vmem = workspace.first_compact_source_vmem
    second_compact_source_vmem = workspace.second_compact_source_vmem
    shared_send_semaphores = workspace.shared_send_semaphores
    shared_receive_semaphores = workspace.shared_receive_semaphores
    output_vmem = workspace.output_vmem
    output_send_semaphores = workspace.output_send_semaphores
    output_receive_semaphores = workspace.output_receive_semaphores
    latent_send_semaphores = workspace.latent_send_semaphores
    latent_receive_semaphores = workspace.latent_receive_semaphores
    shared_local_reduction_vmem = (
        communication_workspace.shared_local_reduction_vmem
    )
    expert_local_reduction_vmem = (
        communication_workspace.expert_local_reduction_vmem
    )
    shared_host_reduction_vmem = (
        communication_workspace.shared_host_reduction_vmem
    )
    expert_host_reduction_vmem = (
        communication_workspace.expert_host_reduction_vmem
    )
    send_semaphores = communication_workspace.send_semaphores
    receive_semaphores = communication_workspace.receive_semaphores
    reduction_phase = communication_workspace.reduction_phase
    latent_vmem = communication_workspace.latent_gather_vmem
    rank = jax.lax.axis_index("tp")
    if synchronize_devices:
        collectives32.barrier()
        reduction_phase[0] = 0
    hidden = hidden_ref[...]

    def copy_expert(expert, slot, *, gate_up=True, down=True):
        return weights.make_expert_copies(
            expert,
            slot,
            gate_up=gate_up,
            down=down,
        )

    if not routing_projection_is_prefetched:
        weights.start_routing_projection_copy()
    latent_output_copy = weights.make_latent_output_copy()
    overlap_latent_gather = selected_expert_count >= 12
    latent_slot = reduction_phase[0] % 2

    def latent_copy(offset):
        view = latent_vmem.at[:, pl.ds(rank * 128, 128)]
        return tpu.make_async_remote_copy(
            view,
            view,
            latent_send_semaphores.at[latent_slot],
            latent_receive_semaphores.at[latent_slot],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def wait_for_latent_gather():
        view = latent_vmem.at[:, : 31 * 128]
        tpu.make_async_copy(
            view, view, latent_send_semaphores.at[latent_slot]
        ).wait()
        tpu.make_async_copy(
            view, view, latent_receive_semaphores.at[latent_slot]
        ).wait()

    def start_latent_gather(latent_input_projection_vmem, latent_input_copy):
        latent_input_copy.wait()
        latent_vmem[:, pl.ds(rank * 128, 128)] = _dot(
            hidden, latent_input_projection_vmem[...]
        ).astype(jnp.bfloat16)
        for offset in (
            31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 19,
            14, 13, 11, 7, 24, 20, 18, 17, 12, 10, 9, 6, 5, 3,
            16, 8, 4, 2, 1,
        ):
            latent_copy(offset).start()

    def prepare_batch(
        latent_input_projection_vmem,
        latent_input_copy_semaphore,
        shared_gate_up_projection_vmem,
        shared_down_projection_vmem,
        dense_copy_semaphores,
    ):
        latent_input_copy = tpu.make_async_copy(
            weights.latent_input_projection,
            latent_input_projection_vmem,
            latent_input_copy_semaphore,
        )
        latent_input_copy.start()
        dense_copies = SimpleNamespace(
            gate_up=tpu.make_async_copy(
                weights.shared_gate_up_projection,
                shared_gate_up_projection_vmem,
                dense_copy_semaphores.at[0],
            ),
            down=tpu.make_async_copy(
                weights.shared_down_projection,
                shared_down_projection_vmem,
                dense_copy_semaphores.at[1],
            ),
        )
        _start_async_copies(dense_copies)

        weights.wait_for_routing_projection_copy()
        shared_route_prefix = jnp.int32(1)
        prefetched_expert_count = jnp.int32(0)

        def prefetch_shared_route_prefix(
            winners,
            owned_locally,
            route_counts,
            unused_selected_experts,
        ):
            del unused_selected_experts
            nonlocal shared_route_prefix, prefetched_expert_count
            shared_route_prefix *= jnp.all(winners == winners[0]).astype(
                jnp.int32
            )
            should_prefetch = (
                shared_route_prefix
                * owned_locally.astype(jnp.int32)[0]
                * (
                    route_counts[0] < expert_prefetch_slot_count
                ).astype(jnp.int32)
            )

            @pl.when(should_prefetch != 0)
            def start_expert_copy():
                _start_async_copies(
                    copy_expert(
                        winners[0] % local_expert_count,
                        route_counts[0],
                    )
                )

            prefetched_expert_count += should_prefetch

        def overlap_routing_with_latent_gather(selection):
            if selection == 3 and overlap_latent_gather:
                start_latent_gather(
                    latent_input_projection_vmem,
                    latent_input_copy,
                )

        (
            selected_experts,
            selected_probabilities,
            local_route_counts,
            selected_probability_totals,
        ) = _route_hidden_states(
            hidden,
            weights.routing_projection_vmem,
            weights.routing_bias,
            workspace.routing_scores_vmem,
            send_semaphores.at[0],
            receive_semaphores.at[0],
            selected_expert_count=selected_expert_count,
            local_expert_count=local_expert_count,
            expert_tensor_parallel=expert_tensor_parallel,
            on_local_selection=prefetch_shared_route_prefix,
            after_selection=overlap_routing_with_latent_gather,
        )
        selected_experts_vmem[...] = jnp.pad(
            selected_experts,
            ((0, 0), (0, 128 - selected_expert_count)),
        )
        selected_probabilities_vmem[...] = jnp.pad(
            selected_probabilities,
            ((0, 0), (0, 128 - selected_expert_count)),
        )
        local_route_counts_vmem[...] = jnp.broadcast_to(
            local_route_counts[:, None], local_route_counts_vmem.shape
        )
        probability_totals_vmem[...] = jnp.broadcast_to(
            selected_probability_totals[:, None],
            probability_totals_vmem.shape,
        )
        _start_initial_expert_weight_copies(
            selected_experts,
            local_route_counts,
            copy_expert,
            slot_count=expert_prefetch_slot_count,
            already_started=prefetched_expert_count,
            group_pending_routes=group_pending_expert_routes,
        )

        if not overlap_latent_gather:
            start_latent_gather(
                latent_input_projection_vmem,
                latent_input_copy,
            )
            wait_for_latent_gather()

        dense_copies.gate_up.wait()
        shared_gate_up = _dot(
            hidden, shared_gate_up_projection_vmem[...]
        ).astype(jnp.bfloat16).astype(jnp.float32)
        middle = _gated_activation(
            shared_gate_up[:, :shared_width],
            shared_gate_up[:, shared_width:],
        )
        dense_copies.down.wait()
        shared_outputs_vmem[...] = _dot(
            middle, shared_down_projection_vmem[...]
        )

    pl.run_scoped(
        prepare_batch,
        tpu.VMEM(
            weights.latent_input_projection.shape,
            weights.latent_input_projection.dtype,
        ),
        tpu.SemaphoreType.DMA,
        tpu.VMEM(
            weights.shared_gate_up_projection.shape,
            weights.shared_gate_up_projection.dtype,
        ),
        tpu.VMEM(
            weights.shared_down_projection.shape,
            weights.shared_down_projection.dtype,
        ),
        tpu.SemaphoreType.DMA((2,)),
    )

    shared_rows = ((hidden_size + 1023) // 1024) * 8
    shared_payload = shared_rows // 2
    expert_rows = expert_output_size // 128
    expert_payload = expert_rows // 2
    # Each lane consumes only its 1024-column shared-expert output shard.
    # At B8, reduce-scattering that shard avoids replicating the full 7168
    # columns without changing the existing BF16 collective boundaries.
    reduce_scatter_shared_output = (
        batch_size in (6, 8) and group_pending_expert_routes
    )
    reduce_scatter_routed_output = routed_reduce_scatter
    # At B8 the encoded routed payload is eight contiguous token shards.
    # Reducing one shard per local lane cuts injected bytes before a final
    # local all-gather restores the batch for the latent-up projection.
    shared_shard_width = 1024
    shared_shard_rows = shared_shard_width // 128
    shared_shard_payload = shared_shard_rows // 2

    def encode_shared(value, destination):
        transport_batch_size = value.shape[0]
        packed = tpu.bitcast(
            value.reshape(transport_batch_size * shared_rows, 128).astype(
                jnp.bfloat16
            ),
            jnp.uint32,
        )
        destination[: transport_batch_size * shared_payload, :] = (
            jax.lax.bitcast_convert_type(packed, jnp.float32)
        )

    def decode_shared(value):
        transport_batch_size = value.shape[1] // shared_payload
        decoded = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(decoded, axis=0).reshape(
            transport_batch_size, shared_rows * 128
        )

    def encode_shared_shard(value, destination):
        transport_batch_size = value.shape[0]
        packed = tpu.bitcast(
            value.reshape(
                transport_batch_size * shared_shard_rows, 128
            ).astype(jnp.bfloat16),
            jnp.uint32,
        )
        destination[: transport_batch_size * shared_shard_payload, :] = (
            jax.lax.bitcast_convert_type(packed, jnp.float32)
        )

    def decode_shared_shard(value):
        transport_batch_size = value.shape[1] // shared_shard_payload
        decoded = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(decoded, axis=0).reshape(
            transport_batch_size, shared_shard_width
        )

    def encode_expert(value, destination):
        transport_batch_size = value.shape[0]
        bits = tpu.bitcast(
            value[:, :expert_output_size]
            .reshape(transport_batch_size * expert_rows, 128)
            .astype(jnp.bfloat16),
            jnp.uint32,
        )
        destination[: transport_batch_size * expert_payload, :] = (
            jax.lax.bitcast_convert_type(bits, jnp.float32)
        )

    def decode_expert(value):
        transport_batch_size = value.shape[1] // expert_payload
        values = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(values, axis=0).reshape(
            transport_batch_size, expert_output_size
        )

    def decode_gathered_experts(value):
        values = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return values.reshape(8, expert_output_size)[:batch_size]

    def shared_copy(payload, offset, across_hosts=False):
        lane, host = rank % 8, rank // 8
        view = (
            shared_host_reduction_vmem.at[host, :payload, :]
            if across_hosts
            else shared_local_reduction_vmem.at[lane, :payload, :]
        )
        semaphore = 6 + offset if across_hosts else offset - 1
        peer = rank ^ (8 * offset if across_hosts else offset)
        return tpu.make_async_remote_copy(
            view,
            view,
            shared_send_semaphores.at[semaphore],
            shared_receive_semaphores.at[semaphore],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def shared_shard_send_view(target_lane):
        # Reuse the expert host buffer until the routed local reduction
        # finishes and needs it for its normal role.
        send_rows = (
            expert_host_reduction_vmem.shape[0]
            * expert_host_reduction_vmem.shape[1]
            // 8
        )
        return pool_alias.view(
            expert_host_reduction_vmem,
            (8, send_rows, 128),
            jnp.float32,
        ).at[target_lane, : batch_size * shared_shard_payload, :]

    def shared_shard_receive_view(source_lane):
        return shared_local_reduction_vmem.at[
            source_lane, : batch_size * shared_shard_payload, :
        ]

    def shared_shard_copy(offset):
        source_lane = rank % 8
        target_lane = source_lane ^ offset
        return tpu.make_async_remote_copy(
            shared_shard_send_view(target_lane),
            shared_shard_receive_view(source_lane),
            shared_send_semaphores.at[offset - 1],
            shared_receive_semaphores.at[offset - 1],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def expert_copy(payload, offset, across_hosts=False):
        view = (
            expert_host_reduction_vmem.at[rank // 8, :payload, :]
            if across_hosts
            else expert_local_reduction_vmem.at[rank % 8, :payload, :]
        )
        semaphore = 6 + offset if across_hosts else offset - 1
        peer = rank ^ (8 * offset if across_hosts else offset)
        return tpu.make_async_remote_copy(
            view,
            view,
            send_semaphores.at[semaphore],
            receive_semaphores.at[semaphore],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def expert_shard_send_view(target_lane):
        return expert_local_reduction_vmem.at[
            rank % 8, : 8 * expert_payload, :
        ].reshape(8, expert_payload, 128).at[target_lane]

    def expert_shard_receive_view(source_lane):
        return expert_local_reduction_vmem.at[
            source_lane, :expert_payload, :
        ]

    def expert_shard_copy(offset):
        source_lane = rank % 8
        target_lane = source_lane ^ offset
        return tpu.make_async_remote_copy(
            expert_shard_send_view(target_lane),
            expert_shard_receive_view(source_lane),
            send_semaphores.at[offset - 1],
            receive_semaphores.at[offset - 1],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def begin_shared_reduction(local_shared_outputs):
        if reduce_scatter_shared_output:
            lane = rank % 8
            for target_lane in range(7):
                encode_shared_shard(
                    local_shared_outputs[
                        :,
                        target_lane * shared_shard_width
                        : (target_lane + 1) * shared_shard_width,
                    ],
                    shared_shard_send_view(target_lane),
                )
            shared_shard_send_view(7)[...] = jnp.zeros(
                (batch_size * shared_shard_payload, 128),
                jnp.float32,
            )
            shared_shard_receive_view(lane)[...] = (
                shared_shard_send_view(lane)[...]
            )
            for offset in range(7, 0, -1):
                shared_shard_copy(offset).start()

        else:
            payload = local_shared_outputs.shape[0] * shared_payload
            assert payload <= shared_local_reduction_vmem.shape[1]
            encode_shared(
                local_shared_outputs,
                shared_local_reduction_vmem.at[rank % 8],
            )
            for offset in range(7, 0, -1):
                shared_copy(payload, offset).start()

    def finish_shared_reduction(payload):
        if reduce_scatter_shared_output:
            for offset in range(7, 0, -1):
                shared_shard_copy(offset).wait()
            encode_shared_shard(
                decode_shared_shard(
                    shared_local_reduction_vmem[:, :payload, :]
                ),
                shared_host_reduction_vmem.at[rank // 8],
            )
        else:
            for offset in range(7, 0, -1):
                shared_copy(payload, offset).wait()
            encode_shared(
                decode_shared(
                    shared_local_reduction_vmem[:, :payload, :]
                ),
                shared_host_reduction_vmem.at[rank // 8],
            )
        for offset in range(3, 0, -1):
            shared_copy(payload, offset, True).start()

    def finish_shared_host_reduction(payload, reduced_shared_outputs):
        for offset in range(3, 0, -1):
            shared_copy(payload, offset, True).wait()
        if reduce_scatter_shared_output:
            lane = rank % 8

            @pl.when(lane < 7)
            def store_shared_shard():
                reduced_shared_outputs[
                    :, pl.ds(lane * shared_shard_width, shared_shard_width)
                ] = decode_shared_shard(
                    shared_host_reduction_vmem[:, :payload, :]
                )
        else:
            reduced_shared_outputs[...] = decode_shared(
                shared_host_reduction_vmem[:, :payload, :]
            )

    def finish_reductions(
        local_expert_outputs,
        reduced_expert_outputs,
        reduced_shared_outputs,
    ):
        transport_batch_size = local_expert_outputs.shape[0]
        shared_transport_payload = transport_batch_size * (
            shared_shard_payload if reduce_scatter_shared_output else shared_payload
        )
        expert_transport_payload = transport_batch_size * expert_payload
        assert shared_transport_payload <= shared_local_reduction_vmem.shape[1]
        assert expert_transport_payload <= expert_local_reduction_vmem.shape[1]

        latent_output_copy.start()
        if reduce_scatter_routed_output:
            routed_scatter_storage = expert_local_reduction_vmem.at[
                rank % 8, : 8 * expert_payload, :
            ]
            if batch_size < 8:
                routed_scatter_storage[...] = jnp.zeros(
                    routed_scatter_storage.shape, jnp.float32
                )
            encode_expert(local_expert_outputs, routed_scatter_storage)
            for offset in range(7, 0, -1):
                expert_shard_copy(offset).start()
        else:
            encode_expert(
                local_expert_outputs,
                expert_local_reduction_vmem.at[rank % 8],
            )
            for offset in range(7, 0, -1):
                expert_copy(expert_transport_payload, offset).start()

        if after_experts is not None:
            after_experts()

        finish_shared_reduction(shared_transport_payload)

        if reduce_scatter_routed_output:
            lane = rank % 8
            for offset in range(7, 0, -1):
                expert_shard_copy(offset).wait()
            owned_shard = expert_shard_send_view(lane)[...]
            expert_shard_receive_view(lane)[...] = owned_shard
            encode_expert(
                decode_expert(
                    expert_local_reduction_vmem[:, :expert_payload, :]
                ),
                expert_host_reduction_vmem.at[rank // 8],
            )
            for offset in range(3, 0, -1):
                expert_copy(expert_payload, offset, True).start()
            for offset in range(3, 0, -1):
                expert_copy(expert_payload, offset, True).wait()

            reduced_token = decode_expert(
                expert_host_reduction_vmem[:, :expert_payload, :]
            )
            encode_expert(
                reduced_token,
                expert_local_reduction_vmem.at[lane],
            )
            for offset in range(7, 0, -1):
                expert_copy(expert_payload, offset).start()
            for offset in range(7, 0, -1):
                expert_copy(expert_payload, offset).wait()
            reduced_expert_outputs[:, :expert_output_size] = (
                decode_gathered_experts(
                    expert_local_reduction_vmem[:, :expert_payload, :]
                )
            )

            finish_shared_host_reduction(
                shared_transport_payload, reduced_shared_outputs
            )
            reduction_phase[0] += 2
            return
        for offset in range(7, 0, -1):
            expert_copy(expert_transport_payload, offset).wait()
        encode_expert(
            decode_expert(
                expert_local_reduction_vmem[:, :expert_transport_payload, :]
            ),
            expert_host_reduction_vmem.at[rank // 8],
        )
        for offset in range(3, 0, -1):
            expert_copy(
                expert_transport_payload, offset, True
            ).start()
        for offset in range(3, 0, -1):
            expert_copy(expert_transport_payload, offset, True).wait()
        reduced_expert_outputs[...] = jnp.zeros(
            reduced_expert_outputs.shape, jnp.float32
        )
        reduced_expert_outputs[:, :expert_output_size] = decode_expert(
            expert_host_reduction_vmem[:, :expert_transport_payload, :]
        )

        finish_shared_host_reduction(
            shared_transport_payload, reduced_shared_outputs
        )
        reduction_phase[0] += 2


    begin_shared_reduction(shared_outputs_vmem[...])
    if overlap_latent_gather:
        wait_for_latent_gather()
    _run_routed_expert_stream(
        latent_vmem,
        selected_experts_vmem[:, :selected_expert_count],
        selected_probabilities_vmem[:, :selected_expert_count],
        local_route_counts_vmem[:, 0],
        probability_totals_vmem[:, 0],
        expert_gate_up_vmem,
        expert_gate_up_scale_vmem,
        expert_down_vmem,
        expert_down_scale_vmem,
        routed_outputs_vmem,
        gate_up_accumulator_vmem,
        down_accumulator_vmem,
        processed_routes_vmem,
        first_compact_source_vmem,
        second_compact_source_vmem,
        copy_expert,
        expert_input_size=expert_input_size,
        expert_width=expert_width,
        expert_output_size=expert_output_size,
        group_pending_routes=group_pending_expert_routes,
    )

    finish_reductions(
        routed_outputs_vmem[...],
        routed_outputs_vmem,
        shared_outputs_vmem,
    )
    routed = routed_outputs_vmem[...].astype(jnp.bfloat16).astype(jnp.float32)
    normalized = (
        routed
        * jax.lax.rsqrt(
            jnp.sum(routed * routed, axis=1, keepdims=True)
            / latent_feature_count
            + epsilon
        )
        * weights.latent_normalization[...]
    ).astype(jnp.bfloat16)
    latent_output_copy.wait()
    # The contraction is independent across batch rows. Let Mosaic lower
    # it as one native row tile so the resident weight is consumed once.
    up = _dot(
        normalized[:, :expert_output_size],
        weights.latent_output_projection_vmem[...],
    )

    local_output = (
        up
        + shared_outputs_vmem[
            :, pl.ds((rank % 8) * 1024, 1024)
        ]
    ).astype(jnp.bfloat16)
    if token_scatter_output:
        # Each lane owns 1024 hidden columns for all eight tokens. Send
        # each peer only the two rows owned by its local rank pair. All
        # seven transfers are independent 4 KiB messages, so the token /
        # feature transpose completes in one communication phase.
        tokens_per_pair = batch_size // 4
        payload_rows = tokens_per_pair * 1024 // 128
        lane = rank % 8
        pair = lane // 2
        send = output_vmem.at[pl.ds(0, 4)]
        receive = output_vmem.at[pl.ds(4, 8)]
        for target_pair in range(4):
            row_start = target_pair * tokens_per_pair
            send[target_pair] = local_output[
                row_start : row_start + tokens_per_pair
            ].reshape(
                payload_rows, 128
            )
        receive[lane] = send[pair]
        copies = []
        for offset in range(7, 0, -1):
            target_lane = lane ^ offset
            target_pair = target_lane // 2
            copy = tpu.make_async_remote_copy(
                send.at[target_pair],
                receive.at[lane],
                output_send_semaphores.at[offset - 1],
                output_receive_semaphores.at[offset - 1],
                device_id=(rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

        for source_lane in range(7):
            block = receive[source_lane].reshape(
                tokens_per_pair, 1024
            )
            output_ref[:, pl.ds(source_lane * 1024, 1024)] = block
    else:
        output_vmem[:, pl.ds((rank % 8) * 1024, 1024)] = local_output
        _gather_output_shards(
            output_vmem,
            output_send_semaphores,
            output_receive_semaphores,
            reduction_phase[0] % 2,
        )
        output_ref[...] = output_vmem[
            :batch_size, :hidden_size
        ].astype(hidden_ref.dtype)
    reduction_phase[0] += 1


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class KimiDeltaAttentionWeights:
    """KDA parameters and their persistent projection-copy resources."""

    gate_projection: TensorLike  # [hidden_size, 128]
    gate_projection_vmem: Any  # [hidden_size, 128]
    input_projection: TensorLike  # [hidden_size, 1664]
    input_projection_vmem: Any  # [hidden_size, 1664]
    feature_gate_projection: TensorLike  # [128, 384]
    feature_gate_projection_vmem: Any  # [128, 384]
    # [384, hidden_size], or [768, hidden_size / 2] after paired-rank repartitioning.
    output_projection: TensorLike
    output_projection_vmem: Any  # [384, hidden_size] or [768, hidden_size / 2]
    convolution_filter: TensorLike  # [4, 3, 3, 128]
    exp_log_decay_rate: TensorLike  # [3, 1]
    time_step_bias: TensorLike  # [3, 128]
    output_normalization: TensorLike  # [128] or [8, 128]
    projection_copy_semaphores: Any  # [4]

    def make_projection_copies(self) -> SimpleNamespace:
        """Create the asynchronous projection-weight copies."""
        return SimpleNamespace(
            gate=tpu.make_async_copy(
                self.gate_projection,
                self.gate_projection_vmem,
                self.projection_copy_semaphores.at[0],
            ),
            feature_gate=tpu.make_async_copy(
                self.feature_gate_projection,
                self.feature_gate_projection_vmem,
                self.projection_copy_semaphores.at[1],
            ),
            input=tpu.make_async_copy(
                self.input_projection,
                self.input_projection_vmem,
                self.projection_copy_semaphores.at[2],
            ),
            output=tpu.make_async_copy(
                self.output_projection,
                self.output_projection_vmem,
                self.projection_copy_semaphores.at[3],
            ),
        )


def kda(
    hidden_ref: TensorLike,  # [batch_size, hidden_size]
    output_ref: TensorLike,  # [batch_size, hidden_size]
    updated_convolution_history_ref: TensorLike,  # [batch_size, 3, 3, 3, 128]
    updated_recurrent_state_ref: TensorLike,  # [batch_size, 3, 128, 128]
    *,
    weights: KimiDeltaAttentionWeights,
    convolution_history_ref: TensorLike,  # [batch_size, 3, 3, 3, 128]
    recurrent_state_ref: TensorLike,  # [batch_size, 3, 128, 128]
    communication_workspace: AttentionCommunicationWorkspace,
    epsilon: float = 1e-5,
    decay_lower_bound: float = -5.0,
    synchronize_devices: bool = True,
    weights_are_prefetched: bool = False,
    token_scatter_output: bool = False,
    sequence_rows: int = 1,
    wait_for_state=None,
    on_state_updated=None,
) -> None:
    """Execute one Kimi Delta Attention layer inside a Pallas program."""
    batch_size, hidden_size = hidden_ref.shape
    head_count, channels_per_head = 3, 128
    state_shape = (
        batch_size,
        head_count,
        channels_per_head,
        channels_per_head,
    )
    history_shape = (batch_size, 3, head_count, 3, channels_per_head)
    if (
        output_ref.shape != hidden_ref.shape
        or weights.gate_projection.shape != (hidden_size, 128)
        or weights.input_projection.shape != (hidden_size, 1664)
        or weights.exp_log_decay_rate.shape != (3, 1)
        or convolution_history_ref.shape != history_shape
        or recurrent_state_ref.shape != state_shape
    ):
        raise ValueError(
            "Kimi Delta Attention inputs and state must match the input batch"
        )
    if batch_size % sequence_rows:
        raise ValueError("sequence_rows must divide the KDA batch")
    if synchronize_devices:
        collectives32.barrier()
        communication_workspace.reduction_phase[0] = 0

    def run(reduction_vmem):
        weight_copies = weights.make_projection_copies()
        if not weights_are_prefetched:
            weight_copies.gate.start()
            weight_copies.feature_gate.start()
            weight_copies.input.start()
            weight_copies.output.start()

        attention_width = head_count * channels_per_head
        weight_copies.gate.wait()
        gate_features = _dot(
            hidden_ref[...], weights.gate_projection_vmem[...]
        ).astype(jnp.bfloat16)

        weight_copies.feature_gate.wait()
        raw_recurrent_gate = (
            _dot(gate_features, weights.feature_gate_projection_vmem[...])
            .astype(jnp.bfloat16)
            .reshape(batch_size, head_count, channels_per_head)
            .astype(jnp.float32)
        )

        weight_copies.input.wait()
        projected = _dot(
            hidden_ref[...], weights.input_projection_vmem[...]
        ).astype(jnp.bfloat16)
        queries_keys_values = projected[:, : 3 * attention_width].reshape(
            batch_size, 3, head_count, channels_per_head
        )
        projected_tail = projected[:, 3 * attention_width :]
        output_gate = projected_tail[:, :attention_width].reshape(
            batch_size, head_count, channels_per_head
        )
        update_rate = (
            projected_tail[:, attention_width : attention_width + head_count]
            .reshape(batch_size, head_count, 1)
            .astype(jnp.float32)
        )
        if wait_for_state is not None:
            wait_for_state()
        decay = decay_lower_bound * jax.nn.sigmoid(
            jnp.exp(weights.exp_log_decay_rate[...])[None]
            * (raw_recurrent_gate + weights.time_step_bias[...][None])
        )
        if sequence_rows == 1:
            convolution_history = convolution_history_ref[...]
            recurrent_state = recurrent_state_ref[...]
            convolution_window = jnp.concatenate(
                (convolution_history, queries_keys_values[:, None]), axis=1
            )
            filtered_queries_keys_values = jax.nn.silu(
                jnp.sum(
                    convolution_window.astype(jnp.float32)
                    * weights.convolution_filter[...].astype(jnp.float32)[None],
                    axis=1,
                )
            ).astype(jnp.bfloat16)
            query, key, value = (
                filtered_queries_keys_values[:, 0].astype(jnp.float32),
                filtered_queries_keys_values[:, 1].astype(jnp.float32),
                filtered_queries_keys_values[:, 2].astype(jnp.float32),
            )
            query *= (
                jax.lax.rsqrt(jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6)
                * channels_per_head**-0.5
            )
            key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
            next_recurrent_state = recurrent_state * jnp.exp(decay)[..., None]
            predicted_value = jnp.sum(key[..., None] * next_recurrent_state, axis=2)
            state_update = jax.nn.sigmoid(update_rate) * (value - predicted_value)
            next_recurrent_state += key[..., None] * state_update[..., None, :]
            attended = jnp.sum(query[..., None] * next_recurrent_state, axis=2)
            updated_convolution_history_ref[...] = convolution_window[:, 1:]
            updated_recurrent_state_ref[...] = next_recurrent_state
        else:
            attended_rows = []
            convolution_filter = weights.convolution_filter[...].astype(jnp.float32)
            for first in range(0, batch_size, sequence_rows):
                window = jnp.concatenate(
                    (
                        convolution_history_ref[first],
                        queries_keys_values[first : first + sequence_rows],
                    ),
                    axis=0,
                )
                state = recurrent_state_ref[first]
                for offset in range(sequence_rows):
                    row = first + offset
                    filtered = jax.nn.silu(
                        jnp.sum(
                            window[offset : offset + 4].astype(jnp.float32)
                            * convolution_filter,
                            axis=0,
                        )
                    ).astype(jnp.bfloat16)
                    query = filtered[0].astype(jnp.float32)
                    key = filtered[1].astype(jnp.float32)
                    value = filtered[2].astype(jnp.float32)
                    query *= (
                        jax.lax.rsqrt(
                            jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6
                        )
                        * channels_per_head**-0.5
                    )
                    key *= jax.lax.rsqrt(
                        jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6
                    )
                    state = state * jnp.exp(decay[row])[..., None]
                    predicted_value = jnp.sum(key[..., None] * state, axis=1)
                    state_update = jax.nn.sigmoid(update_rate[row]) * (
                        value - predicted_value
                    )
                    state = state + key[..., None] * state_update[:, None, :]
                    attended_rows.append(
                        jnp.sum(query[..., None] * state, axis=1)[None]
                    )
                    updated_convolution_history_ref[row] = window[
                        offset + 1 : offset + 4
                    ]
                    updated_recurrent_state_ref[row] = state
            attended = jnp.concatenate(attended_rows, axis=0)
        attended = attended.astype(jnp.bfloat16).astype(jnp.float32)
        attended *= jax.lax.rsqrt(
            jnp.mean(attended * attended, axis=-1, keepdims=True) + epsilon
        )
        normalization = (
            weights.output_normalization[...]
            if len(weights.output_normalization.shape) == 1
            else weights.output_normalization[0]
        )
        attended = (
            (attended * normalization).astype(jnp.bfloat16)
            * jax.nn.sigmoid(output_gate.astype(jnp.float32)).astype(jnp.bfloat16)
        )
        if on_state_updated is not None:
            on_state_updated()

        weight_copies.output.wait()
        _project_attention_2d(
            attended.reshape(batch_size, attention_width),
            weights.output_projection_vmem,
            output_ref,
            reduction_vmem,
            communication_workspace.shared_local_reduction_vmem,
            communication_workspace.shared_host_reduction_vmem,
            communication_workspace.attention_features_vmem,
            communication_workspace.attention_output_vmem,
            communication_workspace.send_semaphores,
            communication_workspace.receive_semaphores,
            communication_workspace.reduction_phase,
            _dot,
            token_scatter_output=token_scatter_output,
        )

    pl.run_scoped(
        run,
        tpu.VMEM((batch_size, 8192), jnp.float32),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class MultiHeadLatentAttentionWeights:
    """MLA parameters and their persistent projection-copy resources."""

    query_down_projection: TensorLike  # [hidden_size, 1536]
    query_down_projection_vmem: Any  # [hidden_size, 1536]
    key_value_down_projection: TensorLike  # [hidden_size, 640]
    key_value_down_projection_vmem: Any  # [hidden_size, 640]
    query_up_projection: TensorLike  # [1536, 768]
    query_up_projection_vmem: Any  # [1536, 768]
    key_value_up_projection: TensorLike  # [512, 768]
    key_value_up_projection_vmem: Any  # [512, 768]
    output_gate_projection: TensorLike  # [hidden_size, 384]
    output_gate_projection_vmem: Any  # [hidden_size, 384]
    # [384, hidden_size], or [768, hidden_size / 2] after paired-rank repartitioning.
    output_projection: TensorLike
    output_projection_vmem: Any  # [384, hidden_size] or [768, hidden_size / 2]
    query_normalization: TensorLike  # [1, 1536]
    key_value_normalization: TensorLike  # [1, 512]
    projection_copy_semaphores: Any  # [6]

    def make_projection_copies(self) -> SimpleNamespace:
        """Create the asynchronous projection-weight copies."""
        return SimpleNamespace(
            query_down=tpu.make_async_copy(
                self.query_down_projection,
                self.query_down_projection_vmem,
                self.projection_copy_semaphores.at[0],
            ),
            key_value_down=tpu.make_async_copy(
                self.key_value_down_projection,
                self.key_value_down_projection_vmem,
                self.projection_copy_semaphores.at[1],
            ),
            query_up=tpu.make_async_copy(
                self.query_up_projection,
                self.query_up_projection_vmem,
                self.projection_copy_semaphores.at[2],
            ),
            key_value_up=tpu.make_async_copy(
                self.key_value_up_projection,
                self.key_value_up_projection_vmem,
                self.projection_copy_semaphores.at[3],
            ),
            output_gate=tpu.make_async_copy(
                self.output_gate_projection,
                self.output_gate_projection_vmem,
                self.projection_copy_semaphores.at[4],
            ),
            output=tpu.make_async_copy(
                self.output_projection,
                self.output_projection_vmem,
                self.projection_copy_semaphores.at[5],
            ),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class MultiHeadLatentAttentionWorkspace:
    """Persistent VMEM and semaphores used by one MLA invocation."""

    key_cache_vmem: Any  # [2, cache_tile_rows, 3, 128, 256]
    value_cache_vmem: Any  # [2, cache_tile_rows, 3, 128, 128]
    cache_copy_semaphores: Any  # [2, 2]
    query_vmem: Any  # [batch_size, 3, 256]
    output_gate_values_vmem: Any  # [batch_size, 384]
    new_key_vmem: Any  # [batch_size, 3, 256]
    new_value_vmem: Any  # [batch_size, 3, 128]
    maximum_vmem: Any  # [batch_size, 3, 1]
    denominator_vmem: Any  # [batch_size, 3, 1]
    accumulator_vmem: Any  # [batch_size, 3, 128]


def mla(
    position_ref: TensorLike,  # [batch_size]
    hidden_ref: TensorLike,  # [batch_size, hidden_size]
    output_ref: TensorLike,  # [batch_size, hidden_size]
    new_key_ref: TensorLike,  # [batch_size, padded_head_count, 256]
    new_value_ref: TensorLike,  # [batch_size, padded_head_count, 128]
    *,
    weights: MultiHeadLatentAttentionWeights,
    key_cache_ref: TensorLike,  # [sequence_count, 3, context_length, 256]
    value_cache_ref: TensorLike,  # [sequence_count, 3, context_length, 128]
    workspace: MultiHeadLatentAttentionWorkspace,
    communication_workspace: AttentionCommunicationWorkspace,
    updated_key_cache_ref: TensorLike | None = None,
    updated_value_cache_ref: TensorLike | None = None,
    epsilon: float = 1e-5,
    synchronize_devices: bool = True,
    weights_are_prefetched: bool = False,
    token_scatter_output: bool = False,
    cache_batch_tile_size: int | None = None,
    sequence_rows: int = 1,
    probability_value_mxu: Literal["bf16", "hilo"] | None = None,
) -> None:
    """Execute one multi-head latent-attention layer inside a Pallas program.

    Consecutive groups of ``sequence_rows`` rows belong to one sequence and
    share one cache entry. Within a group, rows represent consecutive tokens
    and attend causally to earlier rows from the same invocation.
    """
    batch_size, hidden_size = hidden_ref.shape
    context_length = key_cache_ref.shape[2]
    if not 1 <= batch_size <= 8:
        raise ValueError("MLA supports one to eight batch rows")
    if output_ref.shape != hidden_ref.shape:
        raise ValueError("MLA output must match the hidden-state shape")
    if probability_value_mxu not in (None, "bf16", "hilo"):
        raise ValueError("MLA probability-value MXU mode must be 'bf16' or 'hilo'")
    if not 1 <= sequence_rows <= batch_size or batch_size % sequence_rows:
        raise ValueError("sequence_rows must divide the MLA batch")
    sequence_count = batch_size // sequence_rows
    cache_batch_tile_size = cache_batch_tile_size or batch_size
    if not 1 <= cache_batch_tile_size <= batch_size:
        raise ValueError("MLA cache batch tile size must be within the batch")
    if batch_size % cache_batch_tile_size:
        raise ValueError("MLA cache batch tile size must divide the batch")
    if sequence_rows > 1 and cache_batch_tile_size != batch_size:
        raise ValueError("Multi-row sequences do not tile the MLA cache by rows")
    cache_tile_rows = cache_batch_tile_size if sequence_rows == 1 else 1
    first_tile_rows = cache_batch_tile_size if sequence_rows == 1 else sequence_rows
    if position_ref.shape != (batch_size,):
        raise ValueError("Positions must match the MLA batch size")
    if key_cache_ref.shape != (sequence_count, 3, context_length, 256):
        raise ValueError("Key cache must have shape [sequences, 3, context, 256]")
    if value_cache_ref.shape != (sequence_count, 3, context_length, 128):
        raise ValueError("Value cache must have shape [sequences, 3, context, 128]")
    if (
        weights.query_down_projection.shape != (hidden_size, 1536)
        or weights.key_value_down_projection.shape != (hidden_size, 640)
        or weights.query_up_projection.shape != (1536, 768)
        or weights.key_value_up_projection.shape != (512, 768)
        or weights.output_gate_projection.shape != (hidden_size, 384)
        or weights.query_normalization.shape != (1, 1536)
        or weights.key_value_normalization.shape != (1, 512)
        or context_length < 384
        or context_length % 128
    ):
        raise ValueError("MLA projection and normalization shapes are invalid")
    if workspace.key_cache_vmem.shape != (2, cache_tile_rows, 3, 128, 256):
        raise ValueError("MLA key-cache workspace has the wrong shape")
    if workspace.value_cache_vmem.shape != (2, cache_tile_rows, 3, 128, 128):
        raise ValueError("MLA value-cache workspace has the wrong shape")
    expected_workspace_shapes = (
        (workspace.query_vmem.shape, (batch_size, 3, 256)),
        (workspace.output_gate_values_vmem.shape, (batch_size, 384)),
        (workspace.new_key_vmem.shape, (batch_size, 3, 256)),
        (workspace.new_value_vmem.shape, (batch_size, 3, 128)),
        (workspace.maximum_vmem.shape, (batch_size, 3, 1)),
        (workspace.denominator_vmem.shape, (batch_size, 3, 1)),
        (workspace.accumulator_vmem.shape, (batch_size, 3, 128)),
    )
    if any(actual != expected for actual, expected in expected_workspace_shapes):
        raise ValueError("MLA workspace has an invalid shape")
    if (
        new_key_ref.shape[0] != batch_size
        or new_key_ref.shape[1] < 3
        or new_key_ref.shape[2:] != (256,)
        or new_value_ref.shape[0] != batch_size
        or new_value_ref.shape[1] < 3
        or new_value_ref.shape[2:] != (128,)
    ):
        raise ValueError("MLA key/value outputs have invalid shapes")
    if (updated_key_cache_ref is None) != (updated_value_cache_ref is None):
        raise ValueError("Key and value cache outputs must be provided together")
    if updated_key_cache_ref is not None and (
        updated_key_cache_ref.shape != key_cache_ref.shape
        or updated_value_cache_ref.shape != value_cache_ref.shape
    ):
        raise ValueError("MLA cache outputs must match their input cache shapes")

    if synchronize_devices:
        collectives32.barrier()
        communication_workspace.reduction_phase[0] = 0

    weight_copies = weights.make_projection_copies()
    if not weights_are_prefetched:
        weight_copies.query_down.start()
        weight_copies.key_value_down.start()
    weight_copies.query_up.start()
    weight_copies.key_value_up.start()
    weight_copies.output_gate.start()
    weight_copies.output.start()

    positions = jnp.stack(tuple(position_ref[index] for index in range(batch_size)))

    def make_key_cache_copy(block, batch_offset):
        return tpu.make_async_copy(
            key_cache_ref.at[
                pl.ds(batch_offset, cache_tile_rows),
                :,
                pl.ds(block * 128, 128),
                :,
            ],
            workspace.key_cache_vmem.at[block % 2],
            workspace.cache_copy_semaphores.at[0, block % 2],
        )

    def make_value_cache_copy(block, batch_offset):
        return tpu.make_async_copy(
            value_cache_ref.at[
                pl.ds(batch_offset, cache_tile_rows),
                :,
                pl.ds(block * 128, 128),
                :,
            ],
            workspace.value_cache_vmem.at[block % 2],
            workspace.cache_copy_semaphores.at[1, block % 2],
        )

    def prime_cache(batch_offset, maximum_block):
        for block in range(2):

            @pl.when(block <= maximum_block)
            def prime(block=block):
                make_key_cache_copy(block, batch_offset).start()
                make_value_cache_copy(block, batch_offset).start()

    first_tile_maximum_position = functools.reduce(
        jnp.maximum,
        tuple(positions[index] for index in range(first_tile_rows)),
    )
    prime_cache(0, first_tile_maximum_position // 128)

    weight_copies.query_down.wait()
    query_latent = _dot(
        hidden_ref[...], weights.query_down_projection_vmem[...]
    ).astype(jnp.bfloat16).astype(jnp.float32)
    query_latent = (
        query_latent
        * jax.lax.rsqrt(
            jnp.mean(query_latent * query_latent, axis=1, keepdims=True)
            + epsilon
        )
        * weights.query_normalization[...]
    ).astype(jnp.bfloat16)

    weight_copies.key_value_down.wait()
    compressed = _dot(
        hidden_ref[...], weights.key_value_down_projection_vmem[...]
    ).astype(jnp.bfloat16)
    key_value_latent = compressed[:, :512].astype(jnp.float32)
    key_value_latent = (
        key_value_latent
        * jax.lax.rsqrt(
            jnp.mean(
                key_value_latent * key_value_latent,
                axis=1,
                keepdims=True,
            )
            + epsilon
        )
        * weights.key_value_normalization[...]
    ).astype(jnp.bfloat16)

    weight_copies.query_up.wait()
    workspace.query_vmem[...] = _dot(
        query_latent, weights.query_up_projection_vmem[...]
    ).astype(jnp.bfloat16).reshape(batch_size, 3, 256)

    weight_copies.key_value_up.wait()
    expanded = _dot(
        key_value_latent, weights.key_value_up_projection_vmem[...]
    ).astype(jnp.bfloat16).reshape(batch_size, 3, 256)
    workspace.new_key_vmem[...] = jnp.concatenate(
        (
            expanded[:, :, :128],
            jnp.broadcast_to(
                compressed[:, None, 512:576],
                (batch_size, 3, 64),
            ),
            jnp.zeros((batch_size, 3, 64), jnp.bfloat16),
        ),
        axis=2,
    )
    workspace.new_value_vmem[...] = expanded[:, :, 128:]
    new_key_ref[:, :3] = workspace.new_key_vmem[...]
    new_value_ref[:, :3] = workspace.new_value_vmem[...]

    workspace.maximum_vmem[...] = jnp.full(
        (batch_size, 3, 1), -jnp.inf, jnp.float32
    )
    workspace.denominator_vmem[...] = jnp.zeros(
        (batch_size, 3, 1), jnp.float32
    )
    workspace.accumulator_vmem[...] = jnp.zeros(
        (batch_size, 3, 128), jnp.float32
    )

    def attend_cache_tile_body(tile_index, *, compute_output_gate: bool):
        batch_offset = tile_index * cache_batch_tile_size
        tile_positions = tuple(
            position_ref[batch_offset + index]
            for index in range(cache_batch_tile_size)
        )
        head_positions = jnp.stack(
            tuple(position for position in tile_positions for _ in range(3))
        )
        maximum_block = functools.reduce(jnp.maximum, tile_positions) // 128

        @pl.when(tile_index > 0)
        def prime_current_tile():
            prime_cache(batch_offset, maximum_block)

        head_count = cache_batch_tile_size * 3
        batch_slice = pl.ds(batch_offset, cache_batch_tile_size)

        def wait_for_cache_block(block):
            make_key_cache_copy(block, batch_offset).wait()
            make_value_cache_copy(block, batch_offset).wait()

        def attend_block(block):
            slot = block % 2
            token = jnp.arange(128) + block * 128
            key_mask = (
                jax.lax.broadcasted_iota(
                    jnp.int32, (head_count, 128, 256), 1
                )
                + block * 128
                == head_positions[:, None, None]
            )
            key = jnp.where(
                key_mask,
                workspace.new_key_vmem[batch_slice].reshape(
                    head_count, 256
                )[:, None, :],
                workspace.key_cache_vmem[slot].reshape(
                    head_count, 128, 256
                ),
            )
            value_mask = (
                jax.lax.broadcasted_iota(
                    jnp.int32, (head_count, 128, 128), 1
                )
                + block * 128
                == head_positions[:, None, None]
            )
            value = jnp.where(
                value_mask,
                workspace.new_value_vmem[batch_slice].reshape(
                    head_count, 128
                )[:, None, :],
                workspace.value_cache_vmem[slot].reshape(
                    head_count, 128, 128
                ),
            )

            @pl.when(block + 2 <= maximum_block)
            def refill():
                make_key_cache_copy(block + 2, batch_offset).start()
                make_value_cache_copy(block + 2, batch_offset).start()

            score = (
                jnp.sum(
                    workspace.query_vmem[batch_slice]
                    .reshape(head_count, 256)
                    .astype(jnp.float32)[:, None, :]
                    * key.astype(jnp.float32),
                    axis=2,
                )
                * 192**-0.5
            )
            score = jnp.where(
                token[None, :] <= head_positions[:, None],
                score,
                -jnp.inf,
            )
            previous_maximum = workspace.maximum_vmem[batch_slice].reshape(
                head_count, 1
            )
            maximum = jnp.maximum(
                previous_maximum,
                jnp.max(score, axis=1, keepdims=True),
            )
            old_scale = jnp.exp(previous_maximum - maximum)
            probabilities = jnp.exp(score - maximum)
            if probability_value_mxu is None:
                weighted_values = jnp.sum(
                    probabilities[:, :, None] * value.astype(jnp.float32),
                    axis=1,
                )
            else:
                weighted_values = jnp.stack(
                    tuple(
                        _probability_value_mxu(
                            probabilities[index : index + 1],
                            value[index],
                            probability_value_mxu,
                        )[0]
                        for index in range(head_count)
                    )
                )
            workspace.accumulator_vmem[batch_slice] = (
                workspace.accumulator_vmem[batch_slice]
                * old_scale.reshape(cache_batch_tile_size, 3, 1)
                + weighted_values.reshape(cache_batch_tile_size, 3, 128)
            )
            workspace.denominator_vmem[batch_slice] = (
                workspace.denominator_vmem[batch_slice]
                * old_scale.reshape(cache_batch_tile_size, 3, 1)
                + jnp.sum(
                    probabilities,
                    axis=1,
                    keepdims=True,
                ).reshape(cache_batch_tile_size, 3, 1)
            )
            workspace.maximum_vmem[batch_slice] = maximum.reshape(
                cache_batch_tile_size, 3, 1
            )

        # Resolve the first cache dependency before issuing the independent
        # gate projection, allowing the compiler to interleave both regions.
        wait_for_cache_block(0)
        if compute_output_gate:
            weight_copies.output_gate.wait()
            output_gate_values = _dot(
                hidden_ref[...], weights.output_gate_projection_vmem[...]
            ).astype(jnp.bfloat16)

        # Decode positions are nonnegative, so block zero always exists.
        attend_block(0)
        if compute_output_gate:
            workspace.output_gate_values_vmem[...] = output_gate_values

        @pl.loop(1, maximum_block + 1)
        def attend_remaining_blocks(block):
            wait_for_cache_block(block)
            attend_block(block)

    def attend_sequence_body(sequence, *, compute_output_gate: bool):
        first_row = sequence * sequence_rows
        rows = pl.ds(first_row, sequence_rows)
        row_positions = tuple(
            position_ref[first_row + index] for index in range(sequence_rows)
        )
        position_column = jnp.stack(row_positions)[:, None]
        first_block = row_positions[0] // 128
        last_block = row_positions[-1] // 128
        maximum_block = functools.reduce(jnp.maximum, row_positions) // 128

        @pl.when(sequence > 0)
        def prime_current_sequence():
            prime_cache(sequence, maximum_block)

        def wait_for_cache_block(block):
            make_key_cache_copy(block, sequence).wait()
            make_value_cache_copy(block, sequence).wait()

        def attend_block(block):
            slot = block % 2
            key_tile = workspace.key_cache_vmem.at[slot, 0]
            value_tile = workspace.value_cache_vmem.at[slot, 0]

            # Make this invocation visible in the resident cache tile before
            # any row attends, so later rows see earlier rows in the sequence.
            @pl.when(block >= first_block)
            def substitute_new_rows():
                key_tokens = (
                    jax.lax.broadcasted_iota(
                        jnp.int32, (3, 128, 256), 1
                    )
                    + block * 128
                )
                value_tokens = (
                    jax.lax.broadcasted_iota(
                        jnp.int32, (3, 128, 128), 1
                    )
                    + block * 128
                )
                for index in range(sequence_rows):
                    key_tile[...] = jnp.where(
                        key_tokens == row_positions[index],
                        workspace.new_key_vmem[first_row + index][
                            :3, None, :
                        ],
                        key_tile[...],
                    )
                    value_tile[...] = jnp.where(
                        value_tokens == row_positions[index],
                        workspace.new_value_vmem[first_row + index][
                            :3, None, :
                        ],
                        value_tile[...],
                    )

            # Materialize the tile before this double-buffer slot is refilled.
            key = key_tile[...]
            value = (
                value_tile[...]
                if probability_value_mxu is not None
                else value_tile[...].astype(jnp.float32)
            )

            if updated_key_cache_ref is not None:
                @pl.when((block >= first_block) & (block <= last_block))
                def write_back():
                    block_slice = pl.ds(block * 128, 128)
                    key_write = tpu.make_async_copy(
                        key_tile,
                        updated_key_cache_ref.at[
                            sequence, :, block_slice, :
                        ],
                        workspace.cache_copy_semaphores.at[0, slot],
                    )
                    value_write = tpu.make_async_copy(
                        value_tile,
                        updated_value_cache_ref.at[
                            sequence, :, block_slice, :
                        ],
                        workspace.cache_copy_semaphores.at[1, slot],
                    )
                    key_write.start()
                    value_write.start()
                    key_write.wait()
                    value_write.wait()

            @pl.when(block + 2 <= maximum_block)
            def refill():
                make_key_cache_copy(block + 2, sequence).start()
                make_value_cache_copy(block + 2, sequence).start()

            token = jnp.arange(128) + block * 128
            queries = workspace.query_vmem[rows]
            previous_maximum = workspace.maximum_vmem[rows]
            accumulator = workspace.accumulator_vmem[rows]
            denominator = workspace.denominator_vmem[rows]
            new_maximum = []
            new_accumulator = []
            new_denominator = []
            for head in range(3):
                score = (
                    _attention_scores(queries[:, head, :], key[head])
                    * 192**-0.5
                )
                score = jnp.where(
                    token[None, :] <= position_column,
                    score,
                    -jnp.inf,
                )
                maximum = jnp.maximum(
                    previous_maximum[:, head],
                    jnp.max(score, axis=1, keepdims=True),
                )
                old_scale = jnp.exp(previous_maximum[:, head] - maximum)
                probabilities = jnp.exp(score - maximum)
                if probability_value_mxu is None:
                    weighted_values = jnp.sum(
                        probabilities[:, :, None] * value[head][None],
                        axis=1,
                    )
                else:
                    weighted_values = _probability_value_mxu(
                        probabilities,
                        value[head],
                        probability_value_mxu,
                    )
                new_accumulator.append(
                    accumulator[:, head] * old_scale + weighted_values
                )
                new_denominator.append(
                    denominator[:, head] * old_scale
                    + jnp.sum(probabilities, axis=1, keepdims=True)
                )
                new_maximum.append(maximum)
            workspace.maximum_vmem[rows] = jnp.stack(new_maximum, axis=1)
            workspace.accumulator_vmem[rows] = jnp.stack(
                new_accumulator, axis=1
            )
            workspace.denominator_vmem[rows] = jnp.stack(
                new_denominator, axis=1
            )

        wait_for_cache_block(0)
        if compute_output_gate:
            weight_copies.output_gate.wait()
            output_gate_values = _dot(
                hidden_ref[...], weights.output_gate_projection_vmem[...]
            ).astype(jnp.bfloat16)
        attend_block(0)
        if compute_output_gate:
            workspace.output_gate_values_vmem[...] = output_gate_values

        @pl.loop(1, maximum_block + 1)
        def attend_remaining_blocks(block):
            wait_for_cache_block(block)
            attend_block(block)

    if sequence_rows > 1:
        for sequence in range(sequence_count):
            attend_sequence_body(sequence, compute_output_gate=sequence == 0)
    else:
        attend_cache_tile_body(0, compute_output_gate=True)
        if cache_batch_tile_size != batch_size:

            @pl.loop(1, batch_size // cache_batch_tile_size)
            def attend_cache_tile(tile_index):
                attend_cache_tile_body(
                    tile_index, compute_output_gate=False
                )

    attention = (
        (workspace.accumulator_vmem[...] / workspace.denominator_vmem[...])
        .reshape(batch_size, 384)
        .astype(jnp.bfloat16)
        * jax.nn.sigmoid(
            workspace.output_gate_values_vmem[...].astype(jnp.float32)
        ).astype(jnp.bfloat16)
    )
    weight_copies.output.wait()

    def project_output(reduction_vmem):
        _project_attention_2d(
            attention,
            weights.output_projection_vmem,
            output_ref,
            reduction_vmem,
            communication_workspace.shared_local_reduction_vmem,
            communication_workspace.shared_host_reduction_vmem,
            communication_workspace.attention_features_vmem,
            communication_workspace.attention_output_vmem,
            communication_workspace.send_semaphores,
            communication_workspace.receive_semaphores,
            communication_workspace.reduction_phase,
            _dot,
            token_scatter_output=token_scatter_output,
        )

    pl.run_scoped(
        project_output,
        tpu.VMEM((batch_size, 8192), jnp.float32),
    )


def _project_attention_2d(
    attended, weight, out, comm, local, host_reduction, feature_backing,
    output_backing,
    sends, recvs, phase, dot, *, token_scatter_output=False,
):
    rank=jax.lax.axis_index('tp');slot=phase[0]%2
    batch_size=attended.shape[0]
    width=attended.shape[-1];g=weight.shape[0]//width
    assert g==2
    assert 1 <= batch_size <= 8 and out.shape[0]==batch_size and comm.shape[0]==batch_size
    h=out.shape[-1];half=h//g;nr=32//g
    def compute(features,output):
        view=features.at[:,pl.ds((rank%g)*width,width)]
        view[...] = attended
        for offset in range(g-1,0,-1):
            tpu.make_async_remote_copy(view,view,sends.at[10+slot],recvs.at[10+slot],device_id=(rank^offset,),device_id_type=pl.DeviceIdType.MESH).start()
        comm[...] = jnp.zeros(comm.shape,jnp.float32)
        comm[:,:half] = dot(attended,weight[pl.ds((rank%2)*width,width),:])
        count=features.at[:,: (g-1)*width]
        tpu.make_async_copy(count,count,sends.at[10+slot]).wait()
        tpu.make_async_copy(count,count,recvs.at[10+slot]).wait()
        peer_lane=(rank%2)^1
        comm[:,:half] += dot(features[:,pl.ds(peer_lane*width,width)],weight[pl.ds(peer_lane*width,width),:])
        if token_scatter_output:
            assert batch_size in (4, 8)
            tokens_per_pair = batch_size // 4
            payload_rows = tokens_per_pair * half // 128
            host = rank // 8
            pair = (rank % 8) // 2
            parity = rank % 2

            # Reuse attention/MoE communication storage after the input gather.
            # Keep each asynchronous role in a distinct physical lane. This
            # prevents the compiler from reusing aliased transformed refs
            # before every remote send has retired.
            local_send = local.at[0].bitcast(jnp.bfloat16).at[
                pl.ds(0, 4 * payload_rows), :
            ].reshape(4, payload_rows, 128)
            local_receive = local.at[1].bitcast(jnp.bfloat16).at[
                pl.ds(0, 4 * payload_rows), :
            ].reshape(
                4, payload_rows, 128
            )
            # ``output`` is backed by a native 16-row allocation; only the
            # two rows owned by this local pair are live in this path.
            owned_output = _bf16_rows_view(
                output_backing, tokens_per_pair, h
            )
            for target_pair in range(4):
                if tokens_per_pair == 1:
                    partial = comm[target_pair, :half][None, :]
                else:
                    partial = comm[
                        pl.ds(target_pair * tokens_per_pair, tokens_per_pair),
                        :half,
                    ]
                local_send[target_pair] = partial.astype(
                    jnp.bfloat16
                ).reshape(payload_rows, 128)
            local_receive[pair] = local_send[pair]
            local_copies = []
            for offset in range(3, 0, -1):
                target_pair = pair ^ offset
                copy = tpu.make_async_remote_copy(
                    local_send.at[target_pair],
                    local_receive.at[pair],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ (2 * offset),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                local_copies.append(copy)
            for copy in local_copies:
                copy.wait()

            local_sum = jnp.sum(
                local_receive[...].astype(jnp.float32), axis=0
            )
            # The caller supplies this buffer as the four-lane FP32 host
            # reduction workspace used by MoE after attention.
            host_receive = host_reduction.at[:, pl.ds(0, payload_rows), :]
            host_receive[host] = local_sum
            host_copies = []
            host_view = host_receive.at[host]
            for offset in range(3, 0, -1):
                copy = tpu.make_async_remote_copy(
                    host_view,
                    host_view,
                    sends.at[3 + offset - 1],
                    recvs.at[3 + offset - 1],
                    device_id=(rank ^ (8 * offset),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                host_copies.append(copy)
            for copy in host_copies:
                copy.wait()

            reduced_half = jnp.sum(host_receive[...], axis=0).astype(
                jnp.bfloat16
            )
            pair_send = local.at[2].bitcast(jnp.bfloat16).at[
                pl.ds(0, payload_rows), :
            ]
            pair_receive = local.at[3].bitcast(jnp.bfloat16).at[
                pl.ds(0, payload_rows), :
            ]
            pair_send[...] = reduced_half
            pair_copy = tpu.make_async_remote_copy(
                pair_send,
                pair_receive,
                sends.at[6],
                recvs.at[6],
                device_id=(rank ^ 1,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            pair_copy.start()
            pair_copy.wait()
            sent_half = pair_send[...].reshape(tokens_per_pair, half)
            received_half = pair_receive[...].reshape(tokens_per_pair, half)
            owned_output[:, :half] = jnp.where(
                parity == 0, sent_half, received_half
            )
            owned_output[:, half:] = jnp.where(
                parity == 0, received_half, sent_half
            )
        else:
            rows=((half+255)//256)*2;payload=rows//2
            batched_payload=batch_size*payload
            receive=local.reshape(nr,local.shape[1]*8//nr,128)
            assert batched_payload<=receive.shape[1]
            bits=tpu.bitcast(comm[:,:rows*128].reshape(batch_size*rows,128).astype(jnp.bfloat16),jnp.uint32)
            receive[rank//g,:batched_payload,:] = jax.lax.bitcast_convert_type(bits,jnp.float32)
            view=receive.at[rank//g,:batched_payload,:]
            for offset in range(nr-1,0,-1):
                tpu.make_async_remote_copy(view,view,sends.at[14+slot],recvs.at[14+slot],device_id=(rank^(g*offset),),device_id_type=pl.DeviceIdType.MESH).start()
            count=receive.at[:nr-1,:batched_payload,:]
            tpu.make_async_copy(count,count,sends.at[14+slot]).wait()
            tpu.make_async_copy(count,count,recvs.at[14+slot]).wait()
            values=tpu.bitcast(jax.lax.bitcast_convert_type(receive[:,:batched_payload,:],jnp.uint32),jnp.bfloat16).astype(jnp.float32)
            comm[:,:rows*128] = jnp.sum(values,axis=0).reshape(batch_size,rows*128)
            view=output.at[:,pl.ds((rank%g)*half,half)]
            view[...] = comm[:,:half].astype(jnp.bfloat16)
            for offset in range(g-1,0,-1):
                tpu.make_async_remote_copy(view,view,sends.at[12+slot],recvs.at[12+slot],device_id=(rank^offset,),device_id_type=pl.DeviceIdType.MESH).start()
            count=output.at[:,: (g-1)*half]
            tpu.make_async_copy(count,count,sends.at[12+slot]).wait()
            tpu.make_async_copy(count,count,recvs.at[12+slot]).wait()
            out[...] = output[...]
        phase[0]+=1
    # Persistent addresses cannot alias preceding MoE temporaries on a slow peer.
    # Reserve16x1024 BF16 for features and16x7168 for outputs, including
    # native row padding. Total physical storage is exactly256KiB.
    assert g*width<=1024 and h==7168
    feature_store=_bf16_rows_view(feature_backing,batch_size,1024)
    output_store=_bf16_rows_view(output_backing,batch_size,7168)
    compute(feature_store.at[:,:g*width],output_store)

def _gather_latent_host(x,sends,recvs):
    rank=jax.lax.axis_index('tp')
    view=x.at[:,pl.ds((rank%8)*512,512)]
    copies=[]
    for offset in range(7,0,-1):
        cp=tpu.make_async_remote_copy(view,view,sends.at[offset-1],recvs.at[offset-1],device_id=(rank^offset,),device_id_type=pl.DeviceIdType.MESH)
        cp.start();copies.append(cp)
    for cp in copies:cp.wait()


def _router_copy(weight, buffer, sems, wait, *, initialize_padding=True):
    lane = jax.lax.axis_index("tp") % 2

    @pl.when(lane < 1)
    def full():
        copy = tpu.make_async_copy(
            weight.at[:, pl.ds(lane * 512, 512)], buffer.at[0], sems.at[0]
        )
        copy.wait() if wait else copy.start()

    @pl.when(lane == 1)
    def tail():
        if not wait and initialize_padding:
            buffer[0, :, 384:] = jnp.zeros((weight.shape[0], 128), jnp.bfloat16)
        copy = tpu.make_async_copy(
            weight.at[:, pl.ds(512, 384)], buffer.at[0, :, :384], sems.at[0]
        )
        copy.wait() if wait else copy.start()


def _all_reduce_rows(partial, local, hosts, sends, recvs):
    """TP32 sum of an FP32 ``[batch, hidden]`` partial on every rank.

    The shared-expert transport of the mixture-of-experts operation: the
    partial is rounded to BF16 and packed in pairs into 32-bit words, each
    lane sends its packet to the seven other local lanes, the local sum is
    packed again and exchanged among the four hosts. ``local`` and ``hosts``
    are the wide reduction buffers of the communication workspace
    (``[8, rows, 128]`` and ``[4, rows, 128]`` FP32); semaphores 0..6 and
    7..9 are used and left idle.
    """
    rank = jax.lax.axis_index("tp")
    lane, host = rank % 8, rank // 8
    batch_size, hidden = partial.shape
    rows = batch_size * hidden // 128
    payload = rows // 2
    assert rows % 2 == 0 and payload <= local.shape[1] and payload <= hosts.shape[1]

    def encode(value, destination):
        packed = tpu.bitcast(value.reshape(rows, 128).astype(jnp.bfloat16), jnp.uint32)
        destination[:payload, :] = jax.lax.bitcast_convert_type(packed, jnp.float32)

    def decode(buffer):
        decoded = tpu.bitcast(
            jax.lax.bitcast_convert_type(buffer, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(decoded, axis=0).reshape(batch_size, hidden)

    def exchange(buffer, index, offsets, stride, semaphore_base):
        view = buffer.at[index, :payload, :]
        copies = []
        for offset in offsets:
            copy = tpu.make_async_remote_copy(
                view,
                view,
                sends.at[semaphore_base + offset],
                recvs.at[semaphore_base + offset],
                device_id=(rank ^ (stride * offset),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

    encode(partial, local.at[lane])
    exchange(local, lane, range(7, 0, -1), 1, -1)
    encode(decode(local[:, :payload, :]), hosts.at[host])
    exchange(hosts, host, range(3, 0, -1), 8, 6)
    return decode(hosts[:, :payload, :])


def _gather_local_pair_rows(source, destination, sends, recvs, scratch=None):
    """All-gather one token shard among the four same-parity local pairs."""
    rank = jax.lax.axis_index("tp")
    pair = (rank % 8) // 2
    rows_per_pair = source.shape[0]
    if rows_per_pair == 1:
        assert scratch is not None
        payload_rows = source.shape[1] // 128
        source_wire = scratch.at[pair].bitcast(jnp.bfloat16).at[
            pl.ds(0, payload_rows), :
        ]
        source_wire[...] = source[...].reshape(payload_rows, 128)
        copies = []
        for offset in range(3, 0, -1):
            copy = tpu.make_async_remote_copy(
                source_wire,
                source_wire,
                sends.at[offset - 1],
                recvs.at[offset - 1],
                device_id=(rank ^ (2 * offset),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        destination[...] = jnp.stack(
            tuple(
                scratch.at[owner_pair].bitcast(jnp.bfloat16)[
                    :payload_rows
                ].reshape(source.shape[1])
                for owner_pair in range(4)
            )
        )
        return
    for owner_pair in range(4):
        @pl.when(pair == owner_pair)
        def gather_owner():
            row_slice = pl.ds(
                owner_pair * rows_per_pair, rows_per_pair
            )
            destination[row_slice] = source[...]
            copies = []
            for offset in range(3, 0, -1):
                copy = tpu.make_async_remote_copy(
                    source,
                    destination.at[row_slice],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ (2 * offset),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()


# Repeating-stack composition. This is intentionally in the same source file
# as its Pallas components so future batch-shape changes have one edit surface.

MOE = (
    "router",
    "router_bias",
    "latent_down",
    "latent_up",
    "latent_norm",
    "shared_gu",
    "shared_down",
    "expert_gu",
    "expert_gus",
    "expert_down",
    "expert_ds",
)
KDA = (
    "k_gate",
    "k_projection",
    "k_fb",
    "k_o",
    "k_conv",
    "k_a_log",
    "k_dt",
    "k_norm",
)
MLA = ("m_qa", "m_ka", "m_qb", "m_kb", "m_gate", "m_o", "m_qnorm", "m_knorm")
NORMS = ("attn_res", "mlp_res", "attn_norm", "ffn_norm")
EPILOGUE = ("output_res", "final_norm")
DENSE = ("dense_gu", "dense_down")  # layer 0's MLP, used when the stack includes layer 0
# Switchable kernel scheduling options for performance tuning:
#   mla_cache_write_in_attend  sequence-row MLA: write the cache tiles holding
#                           this step's rows back from the attend loop instead
#                           of a separate reload-merge-commit pass;
#   mla_pv_mxu              multiply MLA probabilities by cached values on the
#                           MXU after converting probabilities to BF16;
#   mla_pv_mxu_hilo         use two MXU passes with a BF16 high/low probability
#                           split for accuracy close to the FP32 vector path.
KERNEL_OPTIONS = (
    "mla_cache_write_in_attend",
    "mla_pv_mxu",
    "mla_pv_mxu_hilo",
)
NAMES = MOE + KDA + MLA + NORMS


def local_shapes(
    layers=93,
    hidden=7168,
    *,
    full_model=False,
    vocab=163840,
    expert_storage="mxfp4",
    split_kda_gate_projection: bool = True,
):
    """Per-rank weight shapes. Layer and expert indices are never replicated/tied."""
    if not split_kda_gate_projection:
        raise ValueError("The clean decoder always uses the split KDA gate projection")
    if not 2 <= layers <= 93:
        raise ValueError("Kimi stack supports a prefix of 2..93 layers")
    m = max(1, layers // 4 + (layers == 93))
    k = layers - (layers // 4 + (layers == 93))
    h = hidden
    shapes = (
        (layers - 1, h, 896),
        (layers - 1, 1, 896),
        (layers - 1, h, 128),
        (layers - 1, 128, h),
        (layers - 1, 1, 4096),
        (layers - 1, h, 512),
        (layers - 1, 256, h),
        (layers - 1, 112, 448, 1536),
        (layers - 1, 112, 112, 1536),
        (layers - 1, 112, 96, 3584),
        (layers - 1, 112, 24, 3584),
        (k, h, 128),
        (k, h, 1664 if split_kda_gate_projection else 1792),
        (k, 128, 384),
        (k, 384, h),
        (k, 4, 3, 3, 128),
        (k, 3, 1),
        (k, 3, 128),
        (k, 8, 128),
        (m, h, 1536),
        (m, h, 640),
        (m, 1536, 768),
        (m, 512, 768),
        (m, h, 384),
        (m, 384, h),
        (m, 1, 1536),
        (m, 1, 512),
        *((layers, 1, h) for _ in range(4)),
    )
    result = dict(zip(NAMES, shapes, strict=True))
    if expert_storage != "mxfp4":
        raise ValueError("Routed experts require resident packed FP4")
    if full_model:
        if vocab % 32:
            raise ValueError("Vocabulary must divide into 32 shards")
        result.update(
            embedding=(vocab // 32, h),
            lm_head=(h, vocab // 32),
            dense_gu=(h, 2304),
            dense_down=(1152, h),
            output_res=(1, h),
            final_norm=(1, h),
        )
    return result





def _residual_mixture_row(
    block_values,
    prefix_value,
    folded,
    active_block_count,
    *,
    epsilon,
    relaxed_reduction_order=False,
):
    """Compute one row of the learned residual mixture."""
    def normalized_scores(values):
        scores = jnp.sum(values * folded[...], axis=1, keepdims=True)
        return scores * jax.lax.rsqrt(
            jnp.mean(values * values, axis=1, keepdims=True) + epsilon
        )

    if relaxed_reduction_order:
        block_values = block_values.astype(jnp.float32)
        prefix_value = prefix_value.astype(jnp.float32)

        # Keeping the hidden-sized prefix separate avoids materializing and
        # reducing a nine-row tensor. This changes FP32 association and is
        # therefore intentionally restricted to the relaxed mode.
        scores = jnp.concatenate(
            (
                normalized_scores(block_values),
                normalized_scores(prefix_value),
            ),
            axis=0,
        )
        values = None
    else:
        values = jnp.concatenate(
            (
                block_values,
                prefix_value,
            ),
            axis=0,
        ).astype(jnp.float32)
        scores = jnp.sum(values * folded[...], axis=1, keepdims=True)
        scores *= jax.lax.rsqrt(
            jnp.mean(values * values, axis=1, keepdims=True) + epsilon
        )

    residual_indices = jax.lax.broadcasted_iota(jnp.int32, (9, 1), 0)
    active = (residual_indices < active_block_count) | (
        residual_indices == 8
    )
    probabilities = jax.nn.softmax(
        jnp.where(active, scores, -jnp.inf), axis=0
    )
    if relaxed_reduction_order:
        mixed = jnp.sum(
            block_values * probabilities[:8], axis=0, keepdims=True
        ) + prefix_value * probabilities[8:9]
    else:
        mixed = jnp.sum(values * probabilities, axis=0, keepdims=True)
    return mixed.astype(jnp.bfloat16)


def _residual_mixture_by_row(
    blocks,
    prefix,
    destination,
    folded,
    active_block_count,
    *,
    batch_size,
    epsilon,
    relaxed_reduction_order=False,
):
    """Compute a residual mixture for a batch-leading input."""
    # Pallas traces and statically unrolls this fixed-size loop, so it does not
    # introduce a runtime loop or prevent Mosaic from scheduling rows together.
    for batch_index in range(batch_size):
        destination[batch_index : batch_index + 1] = _residual_mixture_row(
            blocks[batch_index],
            prefix[batch_index : batch_index + 1],
            folded,
            active_block_count,
            epsilon=epsilon,
            relaxed_reduction_order=relaxed_reduction_order,
        )


def _mla_scoped_scratch(
    local_weights,
    *,
    batch_size: int,
    cache_batch_tile_size: int,
    sequence_rows: int,
):
    """Transient MLA resources allocated once around each layer invocation."""
    cache_tile_rows = cache_batch_tile_size if sequence_rows == 1 else 1
    return (
        *(tpu.VMEM(local_weights[name].shape, local_weights[name].dtype) for name in MLA[2:6]),
        tpu.VMEM((2, cache_tile_rows, 3, 128, 256), jnp.bfloat16),
        tpu.VMEM((2, cache_tile_rows, 3, 128, 128), jnp.bfloat16),
        tpu.SemaphoreType.DMA((2, 2)),
        tpu.VMEM((batch_size, 3, 256), jnp.bfloat16),
        tpu.VMEM((batch_size, 384), jnp.bfloat16),
        tpu.VMEM((batch_size, 3, 256), jnp.bfloat16),
        tpu.VMEM((batch_size, 3, 128), jnp.bfloat16),
        tpu.VMEM((batch_size, 3, 1), jnp.float32),
        tpu.VMEM((batch_size, 3, 1), jnp.float32),
        tpu.VMEM((batch_size, 3, 128), jnp.float32),
    )


def _moe_scoped_scratch(
    local_weights,
    *,
    batch_size: int,
    alias_expert_scales: bool,
    token_scatter_output: bool,
):
    """Transient MoE resources not held in the shared projection/expert pool."""
    gate_up = local_weights["expert_gu"]
    down = local_weights["expert_down"]
    latent_size = local_weights["latent_norm"].shape[-1]
    expert_input_size = (
        gate_up.shape[1] * 8 if gate_up.dtype == jnp.uint32 else gate_up.shape[1]
    )
    expert_width = down.shape[1] * 8 if down.dtype == jnp.uint32 else down.shape[1]
    expert_output_size = down.shape[-1]
    scale_scratch = () if alias_expert_scales else (
        tpu.VMEM((4, *local_weights["expert_gus"].shape[1:]), local_weights["expert_gus"].dtype),
        tpu.VMEM((4, *local_weights["expert_ds"].shape[1:]), local_weights["expert_ds"].dtype),
    )
    output_shape = (
        (12, batch_size // 4 * 1024 // 128, 128)
        if token_scatter_output
        else (batch_size, 8192)
    )
    return (
        tpu.VMEM(local_weights["latent_up"].shape, local_weights["latent_up"].dtype),
        tpu.SemaphoreType.DMA,
        tpu.SemaphoreType.DMA((4, 4)),
        *scale_scratch,
        tpu.VMEM((batch_size, 128), jnp.int32),
        tpu.VMEM((batch_size, 128), jnp.float32),
        tpu.VMEM((batch_size, 128), jnp.int32),
        tpu.VMEM((batch_size, 128), jnp.float32),
        tpu.VMEM((batch_size, 1024), jnp.float32),
        tpu.VMEM((batch_size, 7168), jnp.float32),
        tpu.VMEM((batch_size, latent_size), jnp.float32),
        tpu.VMEM((batch_size, 4 * expert_width), jnp.float32),
        tpu.VMEM((batch_size, expert_output_size), jnp.float32),
        tpu.VMEM((batch_size, 128), jnp.int32),
        tpu.VMEM((1, expert_input_size), jnp.bfloat16),
        tpu.VMEM((1, expert_input_size), jnp.bfloat16),
        tpu.SemaphoreType.DMA((10,)),
        tpu.SemaphoreType.DMA((10,)),
        tpu.VMEM(output_shape, jnp.bfloat16),
        tpu.SemaphoreType.DMA((7,) if token_scatter_output else (2,)),
        tpu.SemaphoreType.DMA((7,) if token_scatter_output else (2,)),
        tpu.SemaphoreType.DMA((2,)),
        tpu.SemaphoreType.DMA((2,)),
    )


def stack(
    prefix,
    embedding,
    weights,
    states,
    positions,
    *,
    batch_size,
    layers=93,
    eps=1e-5,
    residual_reduction_order: Literal["reference", "row", "relaxed"] = "reference",
    mla_cache_batch_tile_size: int | None = None,
    alias_expert_scales: bool = False,
    attention_token_scatter: bool = False,
    moe_token_scatter_output: bool = False,
    moe_routed_reduce_scatter: bool = False,
    sequence_parallel_pre_attention: bool = False,
    gate_first_kda_projection: bool = True,
    sequence_rows: int = 1,
    aux_hidden_layers: tuple[int, ...] = (),
    aux_capture: Literal["layer", "attention", "mixture_input"] = "layer",
    include_layer_zero: bool = False,
    state_row=None,
    kernel_options: frozenset = frozenset(),
):
    """Fuse the decoder layers while keeping a native batch tile in VMEM.

    Layers 1..L-1 by default; with ``include_layer_zero`` the loop starts at
    layer 0 (``prefix`` is then the embedding; its KDA runs on the common
    attention path and a dense MLP streamed through the weight pool replaces
    the mixture of experts), which needs ``weights["dense_gu"]`` and
    ``weights["dense_down"]``. ``state_row`` (a traced int32 scalar, one
    sequence per tile only) selects the row of the KDA input states to read
    for the sequence's first row instead of a pre-selected state copy.
    """
    if not 1 <= batch_size <= 8:
        raise ValueError("The fused decoder stack supports batch sizes from 1 to 8")
    if include_layer_zero and any(name not in weights for name in DENSE):
        raise ValueError("include_layer_zero needs the dense MLP weights")
    select_state_row = state_row is not None
    if select_state_row and sequence_rows != batch_size:
        raise ValueError("state_row selection needs one sequence per tile")
    first_layer = 0 if include_layer_zero else 1
    dense_names = DENSE if include_layer_zero else ()
    kernel_options = frozenset(kernel_options)
    if not kernel_options <= set(KERNEL_OPTIONS):
        raise ValueError(f"unknown kernel options {sorted(kernel_options - set(KERNEL_OPTIONS))}")
    mla_cache_write_in_attend = "mla_cache_write_in_attend" in kernel_options and sequence_rows > 1
    if {"mla_pv_mxu", "mla_pv_mxu_hilo"} <= kernel_options:
        raise ValueError("mla_pv_mxu and mla_pv_mxu_hilo are mutually exclusive")
    mla_probability_value_mxu = (
        "hilo"
        if "mla_pv_mxu_hilo" in kernel_options
        else "bf16" if "mla_pv_mxu" in kernel_options else None
    )
    if not 1 <= sequence_rows <= batch_size or batch_size % sequence_rows:
        raise ValueError("sequence_rows must divide the batch")
    if sequence_rows > 1:
        # Rows of one sequence share one MLA cache entry, so the cache is never
        # tiled by rows; the other batch options act on rows and stay valid.
        mla_cache_batch_tile_size = batch_size
    aux_hidden_layers = tuple(aux_hidden_layers)
    if any(not 2 <= index <= layers - 1 for index in aux_hidden_layers) or len(
        set(aux_hidden_layers)
    ) != len(aux_hidden_layers):
        raise ValueError(
            "aux_hidden_layers are distinct layer ids in 2..layers-1; id L "
            "captures the residual stream entering layer L"
        )
    if aux_capture not in ("layer", "attention", "mixture_input"):
        raise ValueError("aux_capture must be 'layer', 'attention' or 'mixture_input'")
    if aux_capture != "layer" and attention_token_scatter:
        raise ValueError("Attention-point aux capture needs complete prefix rows")
    sequences = batch_size // sequence_rows
    if residual_reduction_order not in ("reference", "row", "relaxed"):
        raise ValueError(
            "residual_reduction_order must be 'reference', 'row', or 'relaxed'"
        )
    mla_cache_batch_tile_size = mla_cache_batch_tile_size or batch_size
    if not 1 <= mla_cache_batch_tile_size <= batch_size:
        raise ValueError("MLA cache batch tile size must be within the batch")
    if batch_size % mla_cache_batch_tile_size:
        raise ValueError("MLA cache batch tile size must divide the batch")
    if attention_token_scatter and batch_size not in (4, 8):
        raise ValueError("Attention token scatter requires batch size 4 or 8")
    if moe_token_scatter_output and not (attention_token_scatter and batch_size == 8):
        raise ValueError(
            "MoE token-scattered output requires B8 attention token scatter"
        )
    if moe_routed_reduce_scatter and batch_size not in (6, 8):
        raise ValueError("MoE routed reduce-scatter requires batch size 6 or 8")
    if sequence_parallel_pre_attention and not moe_token_scatter_output:
        raise ValueError(
            "Sequence-parallel pre-attention requires B8 MoE token scatter"
        )
    tokens_per_pair = batch_size // 4 if attention_token_scatter else 0
    cache_handoff_head_count = max(
        3, 8 // math.gcd(mla_cache_batch_tile_size, 8)
    )
    hidden_size = prefix.shape[-1]
    persist_kda_state = batch_size <= 4
    overlap_kda_state_reads = batch_size in (6, 8)
    overlap_kda_state_writes = batch_size in (6, 8)

    def shape(value):
        return jax.ShapeDtypeStruct(value.shape, value.dtype)

    local_weights = {
        name: jax.ShapeDtypeStruct(value.shape[1:], value.dtype)
        for name, value in weights.items()
    }
    mla_scoped_scratch = _mla_scoped_scratch(
        local_weights,
        batch_size=batch_size,
        cache_batch_tile_size=mla_cache_batch_tile_size,
        sequence_rows=sequence_rows,
    )
    moe_scoped_scratch = _moe_scoped_scratch(
        local_weights,
        batch_size=batch_size,
        alias_expert_scales=alias_expert_scales,
        token_scatter_output=moe_token_scatter_output,
    )
    def body(position_ref, state_row_ref, prefix_ref, embedding_ref, *refs):
        layer_weights = dict(zip(NAMES, refs[: len(NAMES)], strict=True))
        output_residual_ref, final_normalization_ref = refs[
            len(NAMES) : len(NAMES) + len(EPILOGUE)
        ]
        dense_start = len(NAMES) + len(EPILOGUE)
        dense_refs = dict(
            zip(dense_names, refs[dense_start : dense_start + len(dense_names)], strict=True)
        )
        state_start = dense_start + len(dense_names)
        convolution_ref, recurrent_ref, key_ref, value_ref = refs[
            state_start : state_start + 4
        ]
        output_refs = refs[state_start + 4 : state_start + 11]
        (
            output_ref,
            blocks_output_ref,
            convolution_output_ref,
            recurrent_output_ref,
            key_output_ref,
            value_output_ref,
            aux_output_ref,
        ) = output_refs
        (
            prefix_vmem,
            blocks_vmem,
            normalized_vmem,
            attention_vmem,
            mixture_vmem,
            attention_residual_vmem,
            mlp_residual_vmem,
            attention_norm_vmem,
            ffn_norm_vmem,
            router_vmem,
            router_semaphores,
            projection_expert_pool_vmem,
            pool_semaphores,
            norm_semaphores,
            convolution_filter_vmem,
            exp_log_decay_rate_vmem,
            time_step_bias_vmem,
            output_normalization_vmem,
            convolution_history_vmem,
            recurrent_state_vmem,
            auxiliary_semaphores,
            query_norms_vmem,
            key_value_norms_vmem,
            routing_biases_vmem,
            latent_norms_vmem,
            metadata_semaphores,
            owned_rows_vmem,
            aux_semaphore,
            dense_semaphores,
            *common,
        ) = refs[state_start + 11 :]
        communication_workspace = AttentionCommunicationWorkspace(*common)

        # In sequence-row mode the chained recurrence reads only the first
        # row's input state; with a state row the accepted row of the
        # previous step is read directly into that slot.
        selected_state_row = state_row_ref[0] if select_state_row else None

        def kda_state_source(ref, state_index):
            if select_state_row:
                return ref.at[pl.ds(selected_state_row, 1), state_index]
            return ref.at[:, state_index]

        def kda_state_destination(vmem):
            return vmem.at[pl.ds(0, 1)] if select_state_row else vmem

        def pool_views(names, *, experts=False):
            views = []
            offset = 0
            for name in names:
                value = local_weights[name]
                logical_shape = value.shape[1:] if experts else value.shape
                view_shape = (4, *logical_shape) if experts else logical_shape
                size = math.prod(view_shape) * value.dtype.itemsize
                storage_size = pool_alias.storage_size(view_shape, value.dtype)
                views.append(
                    pool_alias.view(
                        projection_expert_pool_vmem.at[
                            pl.ds(offset // 128, size // 128), :
                        ],
                        view_shape,
                        value.dtype,
                    )
                )
                offset += storage_size
            return views

        kda_projection_vmem = pool_views(KDA[:4])
        mla_projection_vmem = pool_views(MLA[:2])
        if alias_expert_scales:
            expert_pool_views = pool_views(
                ("expert_gu", "expert_down", "expert_gus", "expert_ds"),
                experts=True,
            )
            expert_vmem = (
                expert_pool_views[0],
                expert_pool_views[2],
                expert_pool_views[1],
                expert_pool_views[3],
            )
        else:
            expert_vmem = pool_views(
                ("expert_gu", "expert_down"), experts=True
            )
        auxiliary_vmem = (
            convolution_filter_vmem,
            exp_log_decay_rate_vmem,
            time_step_bias_vmem,
            output_normalization_vmem,
            convolution_history_vmem,
            recurrent_state_vmem,
        )
        metadata_vmem = (
            query_norms_vmem,
            key_value_norms_vmem,
            routing_biases_vmem,
            latent_norms_vmem,
        )

        collectives32.barrier()
        common[-2][0] = 0
        prefix_vmem[...] = prefix_ref[...]
        blocks_vmem[...] = jnp.zeros(blocks_vmem.shape, jnp.float32)
        blocks_vmem[:, 0] = embedding_ref[...].astype(jnp.float32)

        def residual(destination, folded, active_block_count):
            _residual_mixture_by_row(
                blocks_vmem[...],
                prefix_vmem[...],
                destination,
                folded,
                active_block_count,
                batch_size=batch_size,
                epsilon=eps,
                relaxed_reduction_order=(
                    residual_reduction_order == "relaxed"
                ),
            )

        def normalize(destination, scale):
            value = destination[...].astype(jnp.float32)
            destination[...] = (
                value
                * jax.lax.rsqrt(
                    jnp.mean(value * value, axis=1, keepdims=True) + eps
                )
                * scale[...]
            ).astype(jnp.bfloat16)

        def prefetch_norms(index):
            destinations = (
                attention_residual_vmem,
                mlp_residual_vmem,
                attention_norm_vmem,
                ffn_norm_vmem,
            )
            for slot, (name, destination) in enumerate(
                zip(NORMS, destinations, strict=True)
            ):
                tpu.make_async_copy(
                    layer_weights[name].at[index],
                    destination,
                    norm_semaphores.at[slot],
                ).start()

        def prefetch_attention(index):
            def prefetch_kda():
                state_index = index - index // 4
                # Semaphore slots follow execution dependency order; the VMEM
                # tuple follows schema order, so permute both together. The B8
                # combined projection does not transfer the redundant gate slice.
                projection_indices = (0, 2, 1, 3)
                for slot, projection_index in enumerate(projection_indices):
                    tpu.make_async_copy(
                        layer_weights[KDA[projection_index]].at[state_index],
                        kda_projection_vmem[projection_index],
                        pool_semaphores.at[slot],
                    ).start()

            def prefetch_mla():
                state_index = index // 4
                for slot, (name, destination) in enumerate(
                    zip(MLA[:2], mla_projection_vmem, strict=True)
                ):
                    tpu.make_async_copy(
                        layer_weights[name].at[state_index],
                        destination,
                        pool_semaphores.at[slot],
                    ).start()

            jax.lax.cond(
                (index % 4 == 3) | (index == 92),
                prefetch_mla,
                prefetch_kda,
            )

        def prefetch_auxiliary(index):
            @pl.when((index % 4 != 3) & (index != 92))
            def prefetch_kda_auxiliary():
                state_index = index - index // 4
                sources = tuple(
                    layer_weights[name].at[state_index] for name in KDA[4:]
                )
                destinations = list(auxiliary_vmem[: len(sources)])
                if persist_kda_state:
                    sources += (
                        kda_state_source(convolution_ref, state_index),
                        kda_state_source(recurrent_ref, state_index),
                    )
                    destinations += [
                        kda_state_destination(auxiliary_vmem[4]),
                        kda_state_destination(auxiliary_vmem[5]),
                    ]
                for slot, (source, destination) in enumerate(
                    zip(sources, destinations, strict=True)
                ):
                    tpu.make_async_copy(
                        source,
                        destination,
                        auxiliary_semaphores.at[slot],
                    ).start()

        def prefetch_metadata(index, *, mixture_metadata=True):
            buffer_slot = index % 2

            @pl.when((index % 4 == 3) | (index == 92))
            def prefetch_mla_norms():
                for slot, name in enumerate(("m_qnorm", "m_knorm")):
                    tpu.make_async_copy(
                        layer_weights[name].at[index // 4],
                        metadata_vmem[slot].at[buffer_slot],
                        metadata_semaphores.at[buffer_slot, slot],
                    ).start()

            if not mixture_metadata:
                return  # layer 0 has no mixture of experts (dense MLP instead)
            for slot, name in enumerate(("router_bias", "latent_norm"), 2):
                tpu.make_async_copy(
                    layer_weights[name].at[index - 1],
                    metadata_vmem[slot].at[buffer_slot],
                    metadata_semaphores.at[buffer_slot, slot],
                ).start()

        def prefetch_router(index, *, initialize_padding=False):
            _router_copy(
                layer_weights["router"].at[index - 1],
                router_vmem,
                router_semaphores,
                False,
                initialize_padding=initialize_padding,
            )

        # The lane-1 projection uses 384 of its 512 MXU columns. This persistent
        # tail is never written elsewhere, so initialize it once rather than on
        # every layer prefetch.
        if include_layer_zero:
            # Layer 0 has no router; layer 1's is prefetched after the dense
            # MLP. Initialize the persistent lane-1 padding once here.
            @pl.when(jax.lax.axis_index("tp") % 2 == 1)
            def initialize_router_padding():
                router_vmem[0, :, 384:] = jnp.zeros(
                    (router_vmem.shape[1], 128), jnp.bfloat16
                )
        else:
            prefetch_router(jnp.int32(1), initialize_padding=True)
        prefetch_norms(jnp.int32(first_layer))
        prefetch_attention(jnp.int32(first_layer))
        prefetch_auxiliary(jnp.int32(first_layer))
        prefetch_metadata(jnp.int32(first_layer), mixture_metadata=not include_layer_zero)

        # Layer 0 (dense MLP) is peeled out of the loop as straight-line code
        # so the loop body stays the routed-expert body without a conditional
        # around the mixture (a control-flow region there costs the scheduler
        # the overlap of the mixture's start and tail with attention work).
        def layer(layer_index, *, dense_layer=False):
            def capture_aux_hidden(source=None, gather_owned_rows=False):
                source = prefix_vmem if source is None else source
                for aux_slot, aux_layer in enumerate(aux_hidden_layers):
                    # Layer id L names the residual stream entering layer L,
                    # which is the prefix after layer L - 1 completes (or, in
                    # the debugging 'attention' mode, after its attention).
                    # The rows go to HBM so the capture costs no VMEM.
                    @pl.when(layer_index == aux_layer - 1)
                    def capture(aux_slot=aux_slot):
                        if gather_owned_rows:
                            # Sequence-parallel layers keep only this pair's
                            # prefix rows; complete them for the capture.
                            _gather_local_pair_rows(
                                owned_rows_vmem,
                                prefix_vmem,
                                communication_workspace.send_semaphores,
                                communication_workspace.receive_semaphores,
                                communication_workspace.shared_local_reduction_vmem,
                            )
                        aux_copy = tpu.make_async_copy(
                            source, aux_output_ref.at[aux_slot], aux_semaphore.at[0]
                        )
                        aux_copy.start()
                        aux_copy.wait()

            norm_copies = [
                tpu.make_async_copy(
                    layer_weights[name].at[layer_index],
                    destination,
                    norm_semaphores.at[index],
                )
                for index, (name, destination) in enumerate((
                    ("attn_res", attention_residual_vmem),
                    ("mlp_res", mlp_residual_vmem),
                    ("attn_norm", attention_norm_vmem),
                    ("ffn_norm", ffn_norm_vmem),
                ))
            ]
            norm_copies[0].wait()
            norm_copies[2].wait()
            if sequence_parallel_pre_attention:
                rank = jax.lax.axis_index("tp")
                pair = (rank % 8) // 2
                for owner_pair in range(4):
                    @pl.when(pair == owner_pair)
                    def prepare_owned_attention_rows():
                        row_offset = owner_pair * tokens_per_pair
                        row_slice = pl.ds(row_offset, tokens_per_pair)
                        mixed_rows = []
                        for local_row in range(tokens_per_pair):
                            global_row = row_offset + local_row
                            mixed_rows.append(
                                _residual_mixture_row(
                                    blocks_vmem[global_row],
                                    prefix_vmem[pl.ds(global_row, 1)],
                                    attention_residual_vmem,
                                    (layer_index - 1) // 12 + 1,
                                    epsilon=eps,
                                    relaxed_reduction_order=(
                                        residual_reduction_order == "relaxed"
                                    ),
                                )
                            )
                        owned_rows_vmem[...] = jnp.concatenate(
                            mixed_rows, axis=0
                        )

                        @pl.when(layer_index % 12 == 0)
                        def start_owned_residual_block():
                            block_mask = (
                                jax.lax.broadcasted_iota(
                                    jnp.int32,
                                    blocks_vmem[row_slice].shape,
                                    1,
                                )
                                == layer_index // 12
                            )
                            blocks_vmem[row_slice] = jnp.where(
                                block_mask,
                                prefix_vmem[row_slice][
                                    :, None, :
                                ].astype(jnp.float32),
                                blocks_vmem[row_slice],
                            )
            else:
                residual(
                    normalized_vmem,
                    attention_residual_vmem,
                    (layer_index - 1) // 12 + 1,
                )

                @pl.when(layer_index % 12 == 0)
                def start_residual_block():
                    block_mask = (
                        jax.lax.broadcasted_iota(
                            jnp.int32, blocks_vmem.shape, 1
                        )
                        == layer_index // 12
                    )
                    blocks_vmem[...] = jnp.where(
                        block_mask,
                        prefix_vmem[...][:, None, :].astype(jnp.float32),
                        blocks_vmem[...],
                    )

            if sequence_parallel_pre_attention:
                normalize(owned_rows_vmem, attention_norm_vmem)
                _gather_local_pair_rows(
                    owned_rows_vmem,
                    normalized_vmem,
                    communication_workspace.send_semaphores,
                    communication_workspace.receive_semaphores,
                    communication_workspace.shared_local_reduction_vmem,
                )
            else:
                normalize(normalized_vmem, attention_norm_vmem)

            def run_delta_attention():
                state_index = layer_index - layer_index // 4
                auxiliary_sources = tuple(
                    layer_weights[name].at[state_index] for name in KDA[4:]
                )
                pending = tuple(
                    tpu.make_async_copy(
                        source,
                        destination,
                        auxiliary_semaphores.at[index],
                    )
                    for index, (source, destination) in enumerate(
                        zip(auxiliary_sources, auxiliary_vmem[:4], strict=True)
                    )
                )
                for copy in pending:
                    copy.wait()

                def execute(
                    history_vmem, state_vmem, state_semaphores, state_reads=()
                ):
                    state_writes = (
                        tpu.make_async_copy(
                            history_vmem,
                            convolution_output_ref.at[:, state_index],
                            state_semaphores.at[0],
                        ),
                        tpu.make_async_copy(
                            state_vmem,
                            recurrent_output_ref.at[:, state_index],
                            state_semaphores.at[1],
                        ),
                    )

                    def wait_for_state():
                        for copy in state_reads:
                            copy.wait()

                    def start_state_writes():
                        for copy in state_writes:
                            copy.start()

                    kda(
                        normalized_vmem,
                        attention_vmem,
                        history_vmem,
                        state_vmem,
                        weights=KimiDeltaAttentionWeights(
                            layer_weights["k_gate"].at[state_index],
                            kda_projection_vmem[0],
                            layer_weights["k_projection"].at[state_index],
                            kda_projection_vmem[1],
                            layer_weights["k_fb"].at[state_index],
                            kda_projection_vmem[2],
                            layer_weights["k_o"].at[state_index],
                            kda_projection_vmem[3],
                            *auxiliary_vmem[:4],
                            pool_semaphores,
                        ),
                        convolution_history_ref=history_vmem,
                        recurrent_state_ref=state_vmem,
                        communication_workspace=communication_workspace,
                        epsilon=eps,
                        synchronize_devices=False,
                        weights_are_prefetched=True,
                        token_scatter_output=attention_token_scatter,
                        sequence_rows=sequence_rows,
                        wait_for_state=(
                            wait_for_state if overlap_kda_state_reads else None
                        ),
                        on_state_updated=(
                            start_state_writes if overlap_kda_state_writes else None
                        ),
                    )
                    if not overlap_kda_state_writes:
                        start_state_writes()
                    if not persist_kda_state:
                        for copy in state_writes:
                            copy.wait()

                if persist_kda_state:
                    state_sources = (
                        kda_state_source(convolution_ref, state_index),
                        kda_state_source(recurrent_ref, state_index),
                    )
                    state_pending = tuple(
                        tpu.make_async_copy(
                            source,
                            destination,
                            auxiliary_semaphores.at[index + 4],
                        )
                        for index, (source, destination) in enumerate(
                            zip(
                                state_sources,
                                (
                                    kda_state_destination(convolution_history_vmem),
                                    kda_state_destination(recurrent_state_vmem),
                                ),
                                strict=True,
                            )
                        )
                    )
                    for copy in state_pending:
                        copy.wait()
                    execute(
                        convolution_history_vmem,
                        recurrent_state_vmem,
                        auxiliary_semaphores,
                    )
                else:
                    def execute_scoped(history_vmem, state_vmem, state_semaphores):
                        state_copies = (
                            tpu.make_async_copy(
                                kda_state_source(convolution_ref, state_index),
                                kda_state_destination(history_vmem),
                                state_semaphores.at[0],
                            ),
                            tpu.make_async_copy(
                                kda_state_source(recurrent_ref, state_index),
                                kda_state_destination(state_vmem),
                                state_semaphores.at[1],
                            ),
                        )
                        for copy in state_copies:
                            copy.start()
                        if not overlap_kda_state_reads:
                            for copy in state_copies:
                                copy.wait()
                        execute(
                            history_vmem, state_vmem, state_semaphores, state_copies
                        )

                    pl.run_scoped(
                        execute_scoped,
                        tpu.VMEM(
                            (batch_size, *states[0].shape[2:]), states[0].dtype
                        ),
                        tpu.VMEM(
                            (batch_size, *states[1].shape[2:]), states[1].dtype
                        ),
                        tpu.SemaphoreType.DMA((2,)),
                    )

            def run_latent_attention():
                state_index = layer_index // 4
                buffer_slot = layer_index % 2
                query_norm_vmem = query_norms_vmem.at[buffer_slot]
                key_value_norm_vmem = key_value_norms_vmem.at[buffer_slot]
                layer_metadata_semaphores = metadata_semaphores.at[buffer_slot]

                def compute(new_key_vmem, new_value_vmem, *workspace):
                    metadata_copies = (
                        tpu.make_async_copy(
                            layer_weights["m_qnorm"].at[state_index],
                            query_norm_vmem,
                            layer_metadata_semaphores.at[0],
                        ),
                        tpu.make_async_copy(
                            layer_weights["m_knorm"].at[state_index],
                            key_value_norm_vmem,
                            layer_metadata_semaphores.at[1],
                        ),
                    )
                    for copy in metadata_copies:
                        copy.wait()
                    mla(
                        position_ref,
                        normalized_vmem,
                        attention_vmem,
                        new_key_vmem,
                        new_value_vmem,
                        weights=MultiHeadLatentAttentionWeights(
                            layer_weights["m_qa"].at[state_index],
                            mla_projection_vmem[0],
                            layer_weights["m_ka"].at[state_index],
                            mla_projection_vmem[1],
                            layer_weights["m_qb"].at[state_index],
                            workspace[0],
                            layer_weights["m_kb"].at[state_index],
                            workspace[1],
                            layer_weights["m_gate"].at[state_index],
                            workspace[2],
                            layer_weights["m_o"].at[state_index],
                            workspace[3],
                            query_norm_vmem,
                            key_value_norm_vmem,
                            pool_semaphores,
                        ),
                        key_cache_ref=key_ref.at[:, state_index],
                        value_cache_ref=value_ref.at[:, state_index],
                        workspace=MultiHeadLatentAttentionWorkspace(
                            *workspace[4:]
                        ),
                        communication_workspace=communication_workspace,
                        updated_key_cache_ref=(
                            key_output_ref.at[:, state_index]
                            if mla_cache_write_in_attend
                            else None
                        ),
                        updated_value_cache_ref=(
                            value_output_ref.at[:, state_index]
                            if mla_cache_write_in_attend
                            else None
                        ),
                        epsilon=eps,
                        synchronize_devices=False,
                        weights_are_prefetched=True,
                        token_scatter_output=attention_token_scatter,
                        cache_batch_tile_size=mla_cache_batch_tile_size,
                        sequence_rows=sequence_rows,
                        probability_value_mxu=mla_probability_value_mxu,
                    )

                    def commit_cache(
                        key_tile_vmem,
                        value_tile_vmem,
                        cache_semaphores,
                    ):
                        # Reuse one VMEM cache tile and one loop body across
                        # the logical batch.
                        def commit_cache_tile_body(tile_index):
                            batch_offset = (
                                tile_index * mla_cache_batch_tile_size
                            )
                            positions = jnp.stack(
                                tuple(
                                    position_ref[batch_offset + row_index]
                                    for row_index in range(
                                        mla_cache_batch_tile_size
                                    )
                                )
                            )
                            cache_reads = []
                            for row_index in range(mla_cache_batch_tile_size):
                                batch_index = batch_offset + row_index
                                position = position_ref[batch_index]
                                cache_reads.extend(
                                    (
                                        tpu.make_async_copy(
                                            key_ref.at[
                                                batch_index,
                                                state_index,
                                                :,
                                                pl.ds(
                                                    position // 128 * 128, 128
                                                ),
                                                :,
                                            ],
                                            key_tile_vmem.at[row_index],
                                            cache_semaphores.at[0, row_index],
                                        ),
                                        tpu.make_async_copy(
                                            value_ref.at[
                                                batch_index,
                                                state_index,
                                                :,
                                                pl.ds(
                                                    position // 128 * 128, 128
                                                ),
                                                :,
                                            ],
                                            value_tile_vmem.at[row_index],
                                            cache_semaphores.at[1, row_index],
                                        ),
                                    )
                                )
                            for copy in cache_reads:
                                copy.start()
                            for copy in cache_reads:
                                copy.wait()

                            key_mask = (
                                jax.lax.broadcasted_iota(
                                    jnp.int32, key_tile_vmem.shape, 2
                                )
                                == positions[:, None, None, None] % 128
                            )
                            value_mask = (
                                jax.lax.broadcasted_iota(
                                    jnp.int32, value_tile_vmem.shape, 2
                                )
                                == positions[:, None, None, None] % 128
                            )
                            key_tile_vmem[...] = jnp.where(
                                key_mask,
                                new_key_vmem[
                                    pl.ds(
                                        batch_offset,
                                        mla_cache_batch_tile_size,
                                    )
                                ][:, :3, None, :],
                                key_tile_vmem[...],
                            )
                            value_tile_vmem[...] = jnp.where(
                                value_mask,
                                new_value_vmem[
                                    pl.ds(
                                        batch_offset,
                                        mla_cache_batch_tile_size,
                                    )
                                ][:, :3, None, :],
                                value_tile_vmem[...],
                            )

                            cache_writes = []
                            for row_index in range(mla_cache_batch_tile_size):
                                batch_index = batch_offset + row_index
                                position = position_ref[batch_index]
                                cache_writes.extend(
                                    (
                                        tpu.make_async_copy(
                                            key_tile_vmem.at[row_index],
                                            key_output_ref.at[
                                                batch_index,
                                                state_index,
                                                :,
                                                pl.ds(
                                                    position // 128 * 128, 128
                                                ),
                                                :,
                                            ],
                                            cache_semaphores.at[0, row_index],
                                        ),
                                        tpu.make_async_copy(
                                            value_tile_vmem.at[row_index],
                                            value_output_ref.at[
                                                batch_index,
                                                state_index,
                                                :,
                                                pl.ds(
                                                    position // 128 * 128, 128
                                                ),
                                                :,
                                            ],
                                            cache_semaphores.at[1, row_index],
                                        ),
                                    )
                                )
                            for copy in cache_writes:
                                copy.start()
                            for copy in cache_writes:
                                copy.wait()

                        def commit_sequence_blocks(sequence):
                            # The rows of one sequence are consecutive
                            # positions, so they touch at most two 128-token
                            # cache blocks. Merge the new rows into each block
                            # and write it back.
                            first_row = sequence * sequence_rows
                            row_positions = tuple(
                                position_ref[first_row + index]
                                for index in range(sequence_rows)
                            )
                            first_block = row_positions[0] // 128
                            last_block = row_positions[-1] // 128
                            for extra in range(2):

                                @pl.when(first_block + extra <= last_block)
                                def commit_block(extra=extra):
                                    block = first_block + extra
                                    block_slice = pl.ds(block * 128, 128)
                                    cache_reads = (
                                        tpu.make_async_copy(
                                            key_ref.at[
                                                sequence, state_index, :, block_slice, :
                                            ],
                                            key_tile_vmem.at[0],
                                            cache_semaphores.at[0, 0],
                                        ),
                                        tpu.make_async_copy(
                                            value_ref.at[
                                                sequence, state_index, :, block_slice, :
                                            ],
                                            value_tile_vmem.at[0],
                                            cache_semaphores.at[1, 0],
                                        ),
                                    )
                                    for copy in cache_reads:
                                        copy.start()
                                    for copy in cache_reads:
                                        copy.wait()
                                    key_tokens = (
                                        jax.lax.broadcasted_iota(
                                            jnp.int32, (3, 128, 256), 1
                                        )
                                        + block * 128
                                    )
                                    value_tokens = (
                                        jax.lax.broadcasted_iota(
                                            jnp.int32, (3, 128, 128), 1
                                        )
                                        + block * 128
                                    )
                                    for index in range(sequence_rows):
                                        key_tile_vmem[0] = jnp.where(
                                            key_tokens == row_positions[index],
                                            new_key_vmem[first_row + index][:3, None, :],
                                            key_tile_vmem[0],
                                        )
                                        value_tile_vmem[0] = jnp.where(
                                            value_tokens == row_positions[index],
                                            new_value_vmem[first_row + index][
                                                :3, None, :
                                            ],
                                            value_tile_vmem[0],
                                        )
                                    cache_writes = (
                                        tpu.make_async_copy(
                                            key_tile_vmem.at[0],
                                            key_output_ref.at[
                                                sequence, state_index, :, block_slice, :
                                            ],
                                            cache_semaphores.at[0, 0],
                                        ),
                                        tpu.make_async_copy(
                                            value_tile_vmem.at[0],
                                            value_output_ref.at[
                                                sequence, state_index, :, block_slice, :
                                            ],
                                            cache_semaphores.at[1, 0],
                                        ),
                                    )
                                    for copy in cache_writes:
                                        copy.start()
                                    for copy in cache_writes:
                                        copy.wait()

                        if sequence_rows > 1:
                            if not mla_cache_write_in_attend:
                                for sequence in range(sequences):
                                    commit_sequence_blocks(sequence)
                        elif mla_cache_batch_tile_size == batch_size:
                            commit_cache_tile_body(0)
                        else:

                            @pl.loop(
                                0,
                                batch_size // mla_cache_batch_tile_size,
                            )
                            def commit_cache_tile(tile_index):
                                commit_cache_tile_body(tile_index)

                    commit_tile_rows = (
                        mla_cache_batch_tile_size if sequence_rows == 1 else 1
                    )
                    pl.run_scoped(
                        commit_cache,
                        tpu.VMEM(
                            (commit_tile_rows, 3, 128, 256),
                            jnp.bfloat16,
                        ),
                        tpu.VMEM(
                            (commit_tile_rows, 3, 128, 128),
                            jnp.bfloat16,
                        ),
                        tpu.SemaphoreType.DMA(
                            (2, commit_tile_rows)
                        ),
                    )

                pl.run_scoped(
                    compute,
                    # Pad the transient handoff so each runtime tile advances
                    # by a multiple of eight vector rows.
                    tpu.VMEM(
                        (batch_size, cache_handoff_head_count, 256),
                        jnp.bfloat16,
                    ),
                    tpu.VMEM(
                        (batch_size, cache_handoff_head_count, 128),
                        jnp.bfloat16,
                    ),
                    *mla_scoped_scratch,
                )

            jax.lax.cond(
                (layer_index % 4 == 3) | (layer_index == 92),
                run_latent_attention,
                run_delta_attention,
            )
            norm_copies[1].wait()
            norm_copies[3].wait()
            if attention_token_scatter:
                rank = jax.lax.axis_index("tp")
                pair = (rank % 8) // 2
                owned_attention_vmem = _bf16_rows_view(
                    communication_workspace.attention_output_vmem,
                    tokens_per_pair,
                    hidden_size,
                )
                owned_normalized_vmem = owned_rows_vmem

                # Each local rank pair owns one B4 row or two B8 rows. Static
                # branches keep source slices aligned while calculations use
                # a purpose-sized workspace.
                if tokens_per_pair == 1:
                    prefix_values = prefix_vmem[...]
                for owner_pair in range(4):
                    @pl.when(pair == owner_pair)
                    def prepare_owned_rows():
                        row_offset = owner_pair * tokens_per_pair
                        if tokens_per_pair == 1:
                            prefix_rows = prefix_values[
                                owner_pair : owner_pair + 1
                            ]
                        else:
                            row_slice = pl.ds(row_offset, tokens_per_pair)
                            prefix_rows = prefix_vmem[row_slice]
                        updated_prefix = jnp.where(
                            layer_index % 12 == 0,
                            owned_attention_vmem[...],
                            prefix_rows + owned_attention_vmem[...],
                        ).astype(jnp.bfloat16)
                        if tokens_per_pair == 1:
                            row_indices = jax.lax.broadcasted_iota(
                                jnp.int32, prefix_vmem.shape, 0
                            )
                            prefix_vmem[...] = jnp.where(
                                row_indices == owner_pair,
                                updated_prefix,
                                prefix_values,
                            )
                        else:
                            prefix_vmem[row_slice] = updated_prefix
                        mixed_rows = []
                        for local_row in range(tokens_per_pair):
                            global_row = row_offset + local_row
                            mixed_rows.append(
                                _residual_mixture_row(
                                    blocks_vmem[global_row],
                                    updated_prefix[
                                        local_row : local_row + 1
                                    ],
                                    mlp_residual_vmem,
                                    layer_index // 12 + 1,
                                    epsilon=eps,
                                    relaxed_reduction_order=(
                                        residual_reduction_order == "relaxed"
                                    ),
                                )
                            )
                        owned_normalized_vmem[...] = jnp.concatenate(
                            mixed_rows, axis=0
                        )
                normalize(owned_normalized_vmem, ffn_norm_vmem)
                _gather_local_pair_rows(
                    owned_normalized_vmem,
                    normalized_vmem,
                    communication_workspace.send_semaphores,
                    communication_workspace.receive_semaphores,
                    communication_workspace.shared_local_reduction_vmem,
                )
            else:
                prefix_vmem[...] = jnp.where(
                    layer_index % 12 == 0,
                    attention_vmem[...],
                    prefix_vmem[...] + attention_vmem[...],
                )
                if aux_capture == "attention":
                    capture_aux_hidden()
                residual(
                    normalized_vmem,
                    mlp_residual_vmem,
                    layer_index // 12 + 1,
                )
                normalize(normalized_vmem, ffn_norm_vmem)

            def after_experts():
                if persist_kda_state:
                    @pl.when((layer_index % 4 != 3) & (layer_index != 92))
                    def drain_state_writes():
                        state_index = layer_index - layer_index // 4
                        state_writes = (
                            tpu.make_async_copy(
                                convolution_history_vmem,
                                convolution_output_ref.at[:, state_index],
                                auxiliary_semaphores.at[0],
                            ),
                            tpu.make_async_copy(
                                recurrent_state_vmem,
                                recurrent_output_ref.at[:, state_index],
                                auxiliary_semaphores.at[1],
                            ),
                        )
                        for copy in state_writes:
                            copy.wait()

                @pl.when(layer_index + 1 < layers)
                def prefetch_next_layer():
                    prefetch_norms(layer_index + 1)
                    prefetch_attention(layer_index + 1)
                    prefetch_auxiliary(layer_index + 1)
                    prefetch_metadata(layer_index + 1)
                    prefetch_router(layer_index + 1)

            def run_mixture(*resources):
                buffer_slot = layer_index % 2
                bias_vmem = routing_biases_vmem.at[buffer_slot]
                latent_norm_vmem = latent_norms_vmem.at[buffer_slot]
                metadata_copies = (
                    tpu.make_async_copy(
                        layer_weights["router_bias"].at[layer_index - 1],
                        bias_vmem,
                        metadata_semaphores.at[buffer_slot, 2],
                    ),
                    tpu.make_async_copy(
                        layer_weights["latent_norm"].at[layer_index - 1],
                        latent_norm_vmem,
                        metadata_semaphores.at[buffer_slot, 3],
                    ),
                )
                for copy in metadata_copies:
                    copy.wait()

                latent_output_vmem, latent_output_semaphore, expert_semaphores = (
                    resources[:3]
                )
                if alias_expert_scales:
                    gate_up_vmem, gate_up_scales_vmem, down_vmem, down_scales_vmem = expert_vmem
                    workspace_resources = resources[3:]
                else:
                    gate_up_vmem, down_vmem = expert_vmem
                    gate_up_scales_vmem, down_scales_vmem = resources[3:5]
                    workspace_resources = resources[5:]
                mixture_output_vmem = (
                    owned_rows_vmem if moe_token_scatter_output else mixture_vmem
                )
                moe(
                    normalized_vmem,
                    mixture_output_vmem,
                    weights=MixtureOfExpertsWeights(
                        layer_weights["router"].at[layer_index - 1],
                        router_vmem,
                        router_semaphores,
                        bias_vmem,
                        layer_weights["latent_down"].at[layer_index - 1],
                        layer_weights["latent_up"].at[layer_index - 1],
                        latent_output_vmem,
                        latent_output_semaphore,
                        latent_norm_vmem,
                        layer_weights["shared_gu"].at[layer_index - 1],
                        layer_weights["shared_down"].at[layer_index - 1],
                        layer_weights["expert_gu"].at[layer_index - 1],
                        gate_up_vmem,
                        layer_weights["expert_gus"].at[layer_index - 1],
                        gate_up_scales_vmem,
                        layer_weights["expert_down"].at[layer_index - 1],
                        down_vmem,
                        layer_weights["expert_ds"].at[layer_index - 1],
                        down_scales_vmem,
                        expert_semaphores,
                    ),
                    workspace=MixtureOfExpertsWorkspace(*workspace_resources),
                    communication_workspace=communication_workspace,
                    epsilon=eps,
                    synchronize_devices=False,
                    routing_projection_is_prefetched=True,
                    group_pending_expert_routes=(
                        batch_size in (6, 8)
                        and residual_reduction_order == "relaxed"
                    ),
                    after_experts=after_experts,
                    token_scatter_output=moe_token_scatter_output,
                    routed_reduce_scatter=moe_routed_reduce_scatter,
                )

            def run_dense_mlp():
                """Layer 0's dense MLP with the weights streamed through the
                pool as contiguous row blocks (four slabs in flight): the
                gate/up projection accumulates over 1024-row K blocks of
                ``dense_gu`` (a column block would be 7168 short DMA
                segments), the down projection over 384-row blocks of
                ``dense_down``; TP32 sum with the shared-expert transport;
                output in the mixture's layout."""
                gu_ref, down_ref = dense_refs["dense_gu"], dense_refs["dense_down"]
                k_block, down_block = 1024, 384
                gu_width = gu_ref.shape[1]
                gu_blocks = hidden_size // k_block
                down_blocks = down_ref.shape[0] // down_block
                assert gu_ref.shape == (gu_blocks * k_block, gu_width)
                assert down_ref.shape == (down_blocks * down_block, hidden_size)
                assert gu_width == 2 * down_blocks * down_block
                slab_rows = max(
                    pool_alias.storage_size((k_block, gu_width), gu_ref.dtype),
                    pool_alias.storage_size((down_block, hidden_size), down_ref.dtype),
                ) // 128
                slabs = 4
                assert slabs * slab_rows <= projection_expert_pool_vmem.shape[0]
                total = gu_blocks + down_blocks

                def slab(index, shape, dtype):
                    return pool_alias.view(
                        projection_expert_pool_vmem.at[
                            pl.ds(index * slab_rows, pool_alias.storage_size(shape, dtype) // 128), :
                        ],
                        shape,
                        dtype,
                    )

                def block_copy(k):
                    s = k % slabs
                    if k < gu_blocks:
                        return tpu.make_async_copy(
                            gu_ref.at[pl.ds(k * k_block, k_block), :],
                            slab(s, (k_block, gu_width), gu_ref.dtype),
                            dense_semaphores.at[s],
                        )
                    r = k - gu_blocks
                    return tpu.make_async_copy(
                        down_ref.at[pl.ds(r * down_block, down_block), :],
                        slab(s, (down_block, hidden_size), down_ref.dtype),
                        dense_semaphores.at[s],
                    )

                for k in range(min(slabs, total)):
                    block_copy(k).start()
                hidden_in = normalized_vmem[...]
                projection, activation, accumulator = None, None, None
                for k in range(total):
                    s = k % slabs
                    block_copy(k).wait()
                    if k < gu_blocks:
                        part = _dot(
                            hidden_in[:, k * k_block : (k + 1) * k_block],
                            slab(s, (k_block, gu_width), gu_ref.dtype)[...],
                        )
                        projection = part if projection is None else projection + part
                        if k == gu_blocks - 1:
                            # The mathematical reference rounds the projections to BF16.
                            half = gu_width // 2
                            activation = _gated_activation(
                                projection[:, :half].astype(jnp.bfloat16),
                                projection[:, half:].astype(jnp.bfloat16),
                            )
                    else:
                        r = k - gu_blocks
                        part = _dot(
                            activation[:, r * down_block : (r + 1) * down_block],
                            slab(s, (down_block, hidden_size), down_ref.dtype)[...],
                        )
                        accumulator = part if accumulator is None else accumulator + part
                    if k + slabs < total:
                        block_copy(k + slabs).start()
                # The pool is free again: stage layer 1 while the sum travels.
                after_experts()
                reduced = _all_reduce_rows(
                    accumulator,
                    communication_workspace.shared_local_reduction_vmem,
                    communication_workspace.shared_host_reduction_vmem,
                    communication_workspace.send_semaphores,
                    communication_workspace.receive_semaphores,
                )
                if moe_token_scatter_output:
                    rank = jax.lax.axis_index("tp")
                    pair = (rank % 8) // 2
                    for owner_pair in range(4):
                        @pl.when(pair == owner_pair)
                        def store_owned_dense_rows():
                            owned_rows_vmem[...] = reduced[
                                owner_pair * tokens_per_pair : (owner_pair + 1) * tokens_per_pair
                            ].astype(jnp.bfloat16)
                else:
                    mixture_vmem[...] = reduced.astype(jnp.bfloat16)

            if aux_capture == "mixture_input":
                capture_aux_hidden(normalized_vmem)
            if dense_layer:
                run_dense_mlp()
            else:
                pl.run_scoped(run_mixture, *moe_scoped_scratch)
            if attention_token_scatter:
                rank = jax.lax.axis_index("tp")
                pair = (rank % 8) // 2
                owned_prefix_vmem = owned_rows_vmem
                scattered_mixture = (
                    owned_rows_vmem[...] if moe_token_scatter_output else None
                )
                if tokens_per_pair == 1:
                    prefix_values = prefix_vmem[...]
                    mixture_values = mixture_vmem[...]
                for owner_pair in range(4):
                    @pl.when(pair == owner_pair)
                    def finish_owned_rows():
                        row_offset = owner_pair * tokens_per_pair
                        if tokens_per_pair == 1:
                            finished_prefix = (
                                prefix_values[
                                    owner_pair : owner_pair + 1
                                ]
                                + mixture_values[
                                    owner_pair : owner_pair + 1
                                ]
                            ).astype(jnp.bfloat16)
                        else:
                            row_slice = pl.ds(row_offset, tokens_per_pair)
                            finished_prefix = (
                                prefix_vmem[row_slice]
                                + (
                                    scattered_mixture
                                    if moe_token_scatter_output
                                    else mixture_vmem[row_slice]
                                )
                            ).astype(jnp.bfloat16)
                        owned_prefix_vmem[:, : hidden_size // 2] = (
                            finished_prefix[:, : hidden_size // 2]
                        )
                        owned_prefix_vmem[:, hidden_size // 2 :] = (
                            finished_prefix[:, hidden_size // 2 :]
                        )
                if sequence_parallel_pre_attention:
                    for owner_pair in range(4):
                        @pl.when(pair == owner_pair)
                        def store_owned_prefix_rows():
                            row_offset = owner_pair * tokens_per_pair
                            row_slice = pl.ds(row_offset, tokens_per_pair)
                            prefix_vmem[row_slice] = owned_prefix_vmem[...]
                else:
                    _gather_local_pair_rows(
                        owned_prefix_vmem,
                        prefix_vmem,
                        communication_workspace.send_semaphores,
                        communication_workspace.receive_semaphores,
                        communication_workspace.shared_local_reduction_vmem,
                    )
            else:
                prefix_vmem[...] += mixture_vmem[...]
            if aux_capture == "layer":
                capture_aux_hidden(gather_owned_rows=sequence_parallel_pre_attention)

        if include_layer_zero:
            layer(jnp.int32(0), dense_layer=True)
        pl.loop(1, layers)(layer)

        if sequence_parallel_pre_attention:
            final_weight_copies = (
                tpu.make_async_copy(
                    output_residual_ref,
                    attention_residual_vmem,
                    norm_semaphores.at[0],
                ),
                tpu.make_async_copy(
                    final_normalization_ref,
                    attention_norm_vmem,
                    norm_semaphores.at[2],
                ),
            )
            for copy in final_weight_copies:
                copy.start()
            for copy in final_weight_copies:
                copy.wait()

            rank = jax.lax.axis_index("tp")
            pair = (rank % 8) // 2
            final_block_count = (layers - 1) // 12 + 1
            for owner_pair in range(4):
                @pl.when(pair == owner_pair)
                def finish_owned_decoder_rows():
                    row_offset = owner_pair * tokens_per_pair
                    mixed_rows = []
                    for local_row in range(tokens_per_pair):
                        global_row = row_offset + local_row
                        values = jnp.concatenate(
                            (
                                blocks_vmem[
                                    global_row, :final_block_count
                                ],
                                prefix_vmem[pl.ds(global_row, 1)],
                            ),
                            axis=0,
                        ).astype(jnp.float32)
                        scores = jnp.sum(
                            values * attention_residual_vmem[0], axis=1
                        )
                        scores *= jax.lax.rsqrt(
                            jnp.mean(values * values, axis=1) + eps
                        )
                        mixed_rows.append(
                            jnp.sum(
                                jax.nn.softmax(scores)[:, None] * values,
                                axis=0,
                                keepdims=True,
                            ).astype(jnp.bfloat16)
                        )
                    owned_rows_vmem[...] = jnp.concatenate(
                        mixed_rows, axis=0
                    )
            normalize(owned_rows_vmem, attention_norm_vmem)
            _gather_local_pair_rows(
                owned_rows_vmem,
                normalized_vmem,
                communication_workspace.send_semaphores,
                communication_workspace.receive_semaphores,
                communication_workspace.shared_local_reduction_vmem,
            )
            output_ref[...] = normalized_vmem[...]
            blocks_output_ref[...] = jnp.zeros(
                blocks_output_ref.shape, jnp.uint8
            )
        else:
            output_ref[...] = prefix_vmem[...]
            blocks_output_ref[...] = blocks_vmem[...].astype(jnp.bfloat16)

    args = (
        prefix,
        embedding,
        *(weights[name] for name in NAMES + EPILOGUE + dense_names),
        *states,
    )
    state_row_scalar = (
        jnp.asarray(state_row, jnp.int32).reshape(1)
        if select_state_row
        else jnp.zeros((1,), jnp.int32)
    )
    output_shapes = (
        shape(prefix),
        (
            jax.ShapeDtypeStruct((8, 128), jnp.uint8)
            if sequence_parallel_pre_attention
            else jax.ShapeDtypeStruct(
                (batch_size, 8, hidden_size), jnp.bfloat16
            )
        ),
        *(shape(state) for state in states),
        jax.ShapeDtypeStruct(
            (max(1, len(aux_hidden_layers)), batch_size, hidden_size), jnp.bfloat16
        ),
    )
    pooled_expert_names = (
        ("expert_gu", "expert_gus", "expert_down", "expert_ds")
        if alias_expert_scales
        else ("expert_gu", "expert_down")
    )
    expert_pool_bytes = sum(
        pool_alias.storage_size(
            (4, *local_weights[name].shape[1:]), local_weights[name].dtype
        )
        for name in pooled_expert_names
    )
    attention_pool_bytes = max(
        sum(
            math.prod(local_weights[name].shape)
            * local_weights[name].dtype.itemsize
            for name in names
        )
        for names in (KDA[:4], MLA[:2])
    )
    pool_bytes = max(expert_pool_bytes, attention_pool_bytes)
    assert pool_bytes % 128 == 0
    scratch = (
        tpu.VMEM(prefix.shape, jnp.bfloat16),
        # Residual snapshots originate as BF16, so FP32 storage preserves their
        # values exactly while avoiding a BF16 unpack on every residual mix.
        # The sequence-parallel path logically owns only two B8 rows per rank,
        # but Mosaic corrupted the snapshots across loop iterations when this
        # physical allocation was reduced to two rows. Keep the safe native
        # eight-row layout while reading and updating only the owned rows.
        tpu.VMEM((batch_size, 8, hidden_size), jnp.float32),
        tpu.VMEM(prefix.shape, jnp.bfloat16),
        tpu.VMEM(prefix.shape, jnp.bfloat16),
        tpu.VMEM(prefix.shape, jnp.bfloat16),
        *(tpu.VMEM((1, hidden_size), weights[name].dtype) for name in NORMS),
        tpu.VMEM((1, hidden_size, 512), jnp.bfloat16),
        tpu.SemaphoreType.DMA((2,)),
        tpu.VMEM((pool_bytes // 128, 128), jnp.uint8),
        tpu.SemaphoreType.DMA((6,)),
        tpu.SemaphoreType.DMA((4,)),
        *(tpu.VMEM(local_weights[name].shape, local_weights[name].dtype) for name in KDA[4:]),
        *(
            (
                tpu.VMEM((batch_size, *states[0].shape[2:]), states[0].dtype),
                tpu.VMEM((batch_size, *states[1].shape[2:]), states[1].dtype),
            )
            if persist_kda_state
            else (
                tpu.VMEM((1,), jnp.uint8),
                tpu.VMEM((1,), jnp.uint8),
            )
        ),
        tpu.SemaphoreType.DMA((6,)),
        *(
            tpu.VMEM((2, *local_weights[name].shape), local_weights[name].dtype)
            for name in ("m_qnorm", "m_knorm", "router_bias", "latent_norm")
        ),
        tpu.SemaphoreType.DMA((2, 4)),
        tpu.VMEM(
            (tokens_per_pair, hidden_size)
            if attention_token_scatter
            else (1,),
            jnp.bfloat16,
        ),
        tpu.SemaphoreType.DMA((1,)),
        tpu.SemaphoreType.DMA((4,)),
        *communication_scratch(batch_size)[:-1],
        tpu.VMEM((batch_size, 4096), jnp.bfloat16),
    )
    state_offset = 4 + len(NAMES) + len(EPILOGUE) + len(dense_names)
    return pl.pallas_call(
        body,
        name="kimi_decoder_stack",
        out_shape=output_shapes,
        input_output_aliases={state_offset + index: index + 2 for index in range(4)},
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=tpu.VMEM if index < 2 else tpu.HBM)
                for index in range(len(args))
            ],
            out_specs=[
                pl.BlockSpec(memory_space=tpu.VMEM),
                pl.BlockSpec(memory_space=tpu.VMEM),
                *(pl.BlockSpec(memory_space=tpu.HBM) for _ in range(5)),
            ],
            scratch_shapes=scratch,
        ),
        compiler_params=tpu.CompilerParams(
            collective_id=21,
            vmem_limit_bytes=64 * 1024**2,
            disable_bounds_checks=True,
            shape_invariant_numerics=True,
        ),
    )(positions, state_row_scalar, *args)




def decode_local(
    weights,
    states,
    tokens: TensorLike,  # [batch_size]
    positions: TensorLike,  # [batch_size]
    *,
    batch_size: int,
    layers: int = 93,
    eps: float = 1e-5,
    residual_reduction_order: Literal["reference", "row", "relaxed"] = "reference",
    mla_cache_batch_tile_size: int | None = None,
    alias_expert_scales: bool = False,
    attention_token_scatter: bool = False,
    moe_routed_reduce_scatter: bool = False,
    moe_token_scatter_output: bool = False,
    sequence_parallel_pre_attention: bool = False,
    gate_first_kda_projection: bool | None = None,
    batched_decoder_boundaries: bool | None = None,
    sequence_rows: int = 1,
    aux_hidden_layers: tuple[int, ...] = (),
    aux_capture: Literal["layer", "attention", "mixture_input"] = "layer",
    state_row=None,
    kernel_options: frozenset = frozenset(),
):
    """Decode one native batch tile with local TP32 weights and caches.

    Layer zero (its KDA and dense MLP) runs inside the fused stack after the
    XLA embedding gather. ``state_row`` (int32 scalar, requires
    ``sequence_rows == batch_size``) names the row of the KDA input states
    to start the sequence from, replacing a pre-selected state copy.

    ``sequence_rows`` groups consecutive rows into one sequence of consecutive
    tokens (speculative verification): KDA and MLA states lead with rows and
    sequences respectively, every row's KDA post-state is returned, and MLA
    caches receive all rows. ``aux_hidden_layers`` additionally returns the
    residual stream entering those layers as ``[len, batch, hidden]``.
    """
    if gate_first_kda_projection is False:
        raise ValueError("The clean decoder always uses the split KDA gate projection")
    gate_first_kda_projection = True
    if batched_decoder_boundaries is None:
        # The embedding gather, the dense MLP and (sequence-parallel) lm_head
        # run on the whole tile instead of row by row; the collectives then
        # carry batch-sized payloads, so the results differ from the per-row
        # path at the ULP level.
        batched_decoder_boundaries = batch_size == 8
    if not 1 <= batch_size <= 8:
        raise ValueError("The native decoder supports batch sizes from 1 to 8")
    aux_hidden_layers = tuple(aux_hidden_layers)

    # Compose true-batched layer primitives without the batch-one fused stack.
    rank = jax.lax.axis_index("tp")
    vocabulary = weights["embedding"].shape[0]

    if batched_decoder_boundaries:
        local_ids = tokens - rank * vocabulary
        local_embeddings = weights["embedding"][
            jnp.clip(local_ids, 0, vocabulary - 1)
        ]
        embedding = kimi.tp_sum(
            jnp.where(
                ((local_ids >= 0) & (local_ids < vocabulary))[:, None],
                local_embeddings.astype(jnp.float32),
                0,
            )
        ).astype(jnp.bfloat16)

    else:
        # Preserve the established embedding collective and normalization layout;
        # their weights are tiny. Batch the weight-bearing KDA after the exact row
        # inputs have been formed.
        def prepare_first_layer_row(_, token):
            local_id = token - rank * vocabulary
            embedding_row = weights["embedding"][
                jnp.clip(local_id, 0, vocabulary - 1)
            ][None]
            embedding_row = kimi.tp_sum(
                jnp.where(
                    (local_id >= 0) & (local_id < vocabulary),
                    embedding_row.astype(jnp.float32),
                    0,
                )
            ).astype(jnp.bfloat16)
            return (), embedding_row[0]

        _, embedding = jax.lax.scan(
            prepare_first_layer_row, (), tokens
        )
    if state_row is not None and sequence_rows != batch_size:
        raise ValueError("state_row needs one sequence per tile")
    # Layer zero runs inside the fused stack and starts from the embedding.
    prefix = embedding
    convolution_states, recurrent_states = states[:2]
    key_caches, value_caches = states[2:]
    (
        prefix,
        blocks,
        convolution_states,
        recurrent_states,
        key_caches,
        value_caches,
        aux_hidden,
    ) = (
        stack(
            prefix,
            embedding,
            weights,
            (
                convolution_states,
                recurrent_states,
                key_caches,
                value_caches,
            ),
            positions,
            batch_size=batch_size,
            layers=layers,
            eps=eps,
            residual_reduction_order=residual_reduction_order,
            mla_cache_batch_tile_size=mla_cache_batch_tile_size,
            alias_expert_scales=alias_expert_scales,
            attention_token_scatter=attention_token_scatter,
            moe_token_scatter_output=moe_token_scatter_output,
            moe_routed_reduce_scatter=moe_routed_reduce_scatter,
            sequence_parallel_pre_attention=sequence_parallel_pre_attention,
            gate_first_kda_projection=gate_first_kda_projection,
            sequence_rows=sequence_rows,
            aux_hidden_layers=aux_hidden_layers,
            aux_capture=aux_capture,
            include_layer_zero=True,
            state_row=state_row,
            kernel_options=kernel_options,
        )
    )

    if sequence_parallel_pre_attention and batched_decoder_boundaries:
        logits = jnp.dot(
            prefix, weights["lm_head"], preferred_element_type=jnp.float32
        )
    elif sequence_parallel_pre_attention:
        def project_decoder_row(_, hidden_row):
            logits_row = jnp.dot(
                hidden_row[None],
                weights["lm_head"],
                preferred_element_type=jnp.float32,
            )
            return (), logits_row[0]

        _, logits = jax.lax.scan(project_decoder_row, (), prefix)
    else:
        active_block_count = (layers - 1) // 12 + 1

        def finish_decoder(_, inputs):
            prefix_row, block_rows = inputs
            hidden_row = kimi.attention_residual(
                prefix_row,
                block_rows[:active_block_count],
                weights["output_res"][0],
                eps,
            )[None]
            hidden_row = kimi.rms(hidden_row, weights["final_norm"], eps)
            logits_row = jnp.dot(
                hidden_row,
                weights["lm_head"],
                preferred_element_type=jnp.float32,
            )
            return (), logits_row[0]

        _, logits = jax.lax.scan(finish_decoder, (), (prefix, blocks))
    new_states = (
        convolution_states,
        recurrent_states,
        key_caches,
        value_caches,
    )
    if aux_hidden_layers:
        return logits[None], aux_hidden, new_states
    return logits[None], new_states


def make_decode(
    mesh,
    *,
    batch_size=1,
    layers=93,
    donate=True,
    residual_reduction_order: Literal["reference", "row", "relaxed"] = "reference",
    mla_cache_batch_tile_size: int | None = None,
    alias_expert_scales: bool = False,
    attention_token_scatter: bool = False,
    moe_token_scatter_output: bool = False,
    moe_routed_reduce_scatter: bool = False,
    sequence_parallel_pre_attention: bool = False,
    gate_first_kda_projection: bool | None = None,
    batched_decoder_boundaries: bool | None = None,
    sequence_rows: int = 1,
    aux_hidden_layers: tuple[int, ...] = (),
    aux_capture: Literal["layer", "attention", "mixture_input"] = "layer",
    select_state_row: bool = False,
    kernel_options: frozenset = frozenset(),
):
    """Global arrays lead with rank; logits are [32, batch_size, V/32].

    With ``select_state_row`` the program takes a fifth argument, the int32
    row of the KDA input states the sequence starts from (one sequence per
    tile), so callers need not select and repeat the states themselves.

    With ``aux_hidden_layers`` the result is ``(logits, aux, states)`` where
    ``aux`` is the replicated ``[len(aux_hidden_layers), batch_size, hidden]``
    residual stream entering each listed layer. With ``sequence_rows > 1`` the
    KDA states lead with ``[32, batch_size, ...]`` rows and the MLA caches with
    ``[32, batch_size // sequence_rows, ...]`` sequences.
    """
    from jax.sharding import PartitionSpec as P

    if gate_first_kda_projection is False:
        raise ValueError("The clean decoder always uses the split KDA gate projection")
    gate_first_kda_projection = True

    if mesh.size != 32:
        raise ValueError("Kimi requires 32 devices (TP32)")
    if not 1 <= batch_size <= 8:
        raise ValueError("Kimi decode currently supports batch sizes from 1 to 8")
    if not 1 <= sequence_rows <= batch_size or batch_size % sequence_rows:
        raise ValueError("sequence_rows must divide the batch")
    aux_hidden_layers = tuple(aux_hidden_layers)
    sequences = batch_size // sequence_rows

    def local(weights, states, token, position, *extra):
        state_row = extra[0] if select_state_row else None
        weights = jax.tree.map(lambda x: x[0], weights)
        states = jax.tree.map(lambda x: x[0], states)
        if token.shape != (batch_size,) or position.shape != (batch_size,):
            raise ValueError("Token and position arrays must match the batch size")
        if any(state.shape[0] != batch_size for state in states[:2]):
            raise ValueError("KDA states must have a leading batch dimension")
        if any(state.shape[0] != sequences for state in states[2:]):
            raise ValueError("MLA caches must have a leading sequence dimension")
        result = decode_local(
            weights,
            states,
            token,
            position,
            batch_size=batch_size,
            layers=layers,
            residual_reduction_order=residual_reduction_order,
            mla_cache_batch_tile_size=mla_cache_batch_tile_size,
            alias_expert_scales=alias_expert_scales,
            attention_token_scatter=attention_token_scatter,
            moe_token_scatter_output=moe_token_scatter_output,
            moe_routed_reduce_scatter=moe_routed_reduce_scatter,
            sequence_parallel_pre_attention=sequence_parallel_pre_attention,
            gate_first_kda_projection=gate_first_kda_projection,
            batched_decoder_boundaries=batched_decoder_boundaries,
            sequence_rows=sequence_rows,
            aux_hidden_layers=aux_hidden_layers,
            aux_capture=aux_capture,
            state_row=state_row,
            kernel_options=frozenset(kernel_options),
        )
        states = jax.tree.map(lambda x: x[None], result[-1])
        if aux_hidden_layers:
            return result[0], result[1], states
        return result[0], states

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("tp"), P("tp"), P(), P()) + ((P(),) if select_state_row else ()),
            out_specs=(P("tp"), P(), P("tp")) if aux_hidden_layers else (P("tp"), P("tp")),
            check_vma=False,
        ),
        donate_argnums=(1,) if donate else (),
    )


def _logical_mesh(device_order_bits: str):
    """Build the 32-device logical TP mesh used by Kimi K3."""
    import numpy as np
    from jax.sharding import Mesh

    devices = sorted(jax.devices(), key=lambda device: (device.process_index, device.id))
    if len(devices) != 32:
        raise RuntimeError(f"Expected 32 TPU devices, found {len(devices)}")
    bits = tuple(map(int, device_order_bits.split(",")))
    if sorted(bits) != list(range(5)) or bits[0] != 0:
        raise ValueError(
            "device_order_bits must permute 0..4 while preserving chip-pair bit 0"
        )
    order = [
        sum(((rank >> bit) & 1) << physical for bit, physical in enumerate(bits))
        for rank in range(32)
    ]
    return Mesh(np.array([devices[index] for index in order]), ("tp",)), bits, order
