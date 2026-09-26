# Muse Spark 1.2 megakernel: design notes for engineers

The `musespark/` package serves Muse Spark 1.2 (816B-A42B) on one host with four
TPU7x chips. This file records how the pieces fit: the sharding, the per-rank
layouts and container formats, what one decode step does inside the kernel, the
VMEM budget, the collectives, the rounding policy and the open items. The
top-level `README.md` covers setup, running and results. Measurements quoted
below name the log they come from; `notes/perf_log.md`,
`notes/nvfp4_feasibility.md` and `notes/nvfp4_container_v2.md` live in the
engineering scratchpad
(`/filestore/tmp/claude-0/-filestore-weights/bc41d5f3-5fda-4ab7-9ca1-48e0c99553cc/scratchpad/notes/`).

## 1. Model and hardware

`Config` (`musespark/config.py`, read from the checkpoint's `config.json`) is the
single source of truth for every shape; the `MINI` config (4 layers, hidden 1024,
16 experts top-4, 16 query / 8 KV heads, vocab 2048, window 256, group 64) runs
the identical kernel code in the tests.

| | real model | notes |
|---|---|---|
| layers `L` | 62 | full-attention (NoPE) layers `l % 4 == 1` (16), sliding-window RoPE layers otherwise (46) |
| hidden `H` / moe hidden `Hm` / expert FFN `I` | 8192 / 4096 / 4096 | `pre_expert_proj` maps H -> Hm, experts Hm -> Hm, `post_expert_proj` Hm -> H |
| experts `E` / top-k | 256 / 8 | sigmoid router with a selection bias, unbiased weights renormalised |
| heads / KV heads / head dim | 128 / 16 / 64 | GQA groups of 8, parameter-free QK norms, RoPE theta 5e5 (rotate-half) |
| sliding window | 2048 | keys per query including the query token |
| vocab / used | 202048 / 201818 | ids >= 201818 are untrained padding rows, never sampled |
| softcap / output multiplier | 20 / 2^-2.5 | on the router logits and the final logits |

Hardware: one host, four TPU7x chips x two TensorCores = eight JAX devices,
`Mesh(devices[:8], ("tp",))`, ~94.7 GB HBM and 64 MiB VMEM per core. Devices
sorted by id: `2r` and `2r + 1` are the two cores of chip `r`, so `rank ^ 1` is
the sibling core. Measured (hwbench `hier_bench.py`, `perf_log.md` E1): a remote
VMEM copy to the sibling runs at 101 GB/s for 256 KiB (2.6 us) up to 610 GB/s for
16 MiB, every cross-chip link at a flat ~45 GB/s per core per direction, the
three links in parallel at 134 GB/s.

## 2. Sharding (TP8 over the eight cores)

Rank `r = lax.axis_index("tp")` owns:

- attention: query heads `16r .. 16r + 15` (two GQA groups) and KV heads `2r, 2r + 1`;
  `o_proj` is input-sharded, so the o-projection yields a partial `[B, H]` sum
  that is all-reduced;
- routed experts: **every** expert, but only its intermediate slice
  `Is = I / 8 = 512`: `gate_up [Hm, 2 * 512]` (this rank's 512 gate columns |
  512 up columns) and `down [512, Hm]`. All ranks therefore process the same
  experts for the same batch rows and their `down` outputs are partial sums,
  reduced by one all-reduce per layer;
- `pre_expert_proj` output-sharded `[H, 512]` and `post_expert_proj`
  output-sharded `[Hm, 1024]`, each followed by an all-gather;
- vocabulary shard `r * 25600 .. (r + 1) * 25600` of the embedding and `lm_head`
  (`Vp = vocab_pad = 25600`);
- replicated: norms, residual-gate coefficients, router (as a bf16 hi/lo pair of
  the f32 matrix), router bias, and the f32 residual stream `s [B, H]`.

All vector work is done redundantly on every rank and must be bit-identical
across ranks, which the collectives guarantee by summing in a fixed order; the
routing (top-k on replicated logits) then agrees on every rank without any
exchange.

## 3. Per-rank layouts (`musespark/layout.py`)

Every matrix is `[in, out]` (`y = x @ W`), layer axis first; the global array has
a leading `tp` axis sharded `P("tp")`. With `qh = 16`, `kvh = 2`, `D = 64`,
`Is = 512`, `KC = min(512, K)` the static K chunk of the in-kernel dots:

| family | per-rank shape | dtype | notes |
|---|---|---|---|
| `embed` | `[25600, 8192]` | bf16 | rows of this rank's vocab shard, zero-padded |
| `lm_head` | `[8192, 25600]` | bf16 | streamed through the ring after the last layer |
| `final_norm` | `[1, 8192]` | bf16 | as-is (gain centred at 0, no +1) |
| `attn_norm`, `ffn_norm`, `post_ffn_norm` | `[62, 1, 8192]` | bf16 | effective `r16(1 + gamma)`; `post_ffn_norm` multiplies BEFORE the norm |
| `attn_gate_alpha/beta`, `ffn_gate_alpha/beta` | `[62, 1, 8192]` | f32 | `s = alpha * s + beta * branch` (`gate_coeffs`) |
| `pre_expert_norm`, `post_expert_norm` | `[62, 1, 4096]` | bf16 | effective weights; `post_expert_norm` multiplies BEFORE the norm |
| `router_bias` | `[62, 1, 256]` | f32 | selection only |
| `q`, `gate` | `[62, 8192, 1024]` | bf16 | 16 heads x 64 lanes, head-major |
| `kv` | `[62, 8192, 256]` | bf16 | lanes 0:128 = the two K heads, 128:256 = the two V heads |
| `o` | `[62, 1024, 8192]` | bf16 | input-sharded -> partial sums |
| `pre` | `[62, 8192, 512]` | bf16 | output-sharded |
| `router_hi`, `router_lo` | `[62, 8192, 256]` | bf16 | `hi = bf16(W)`, `lo = bf16(W - hi)`, `logits = dot(x, hi) + dot(x, lo)` in f32 |
| `post` | `[62, 4096, 1024]` | bf16 | output-sharded |
| int8 dense (`dense_format` int8) `q_i8`, `kv_i8`, `gate_i8`, `o_i8`, `pre_i8`, `post_i8` | the bf16 shapes | int8 | per-output-column `w ~= q * s`, `s = absmax_K / 127` (`quant.quantize_int8`); `o_s` over the full o_proj K (identical on every rank) |
| `q_s`, `kv_s`, `gate_s`, `o_s`, `pre_s`, `post_s` | `[62, 1, N]` | f32 | prefetched per layer like the norm vectors |
| `lm_head_i8`, `lm_head_s` | `[8192, 25600]`, `[1, 25600]` | int8, f32 | padded columns: `q = 0`, `s = 1` |
| int4 (format v1) `gate_up_q` | `[62, 256, 4096, 1024]` | int4 | cols 0:512 gate, 512:1024 up |
| `gate_up_s` | `[62, 256, 8, 4, 1024]` | f32 | group-128 scales, `[K/KC, KC/G, N]` chunked |
| `down_q` | `[62, 256, 512, 4096]` | int4 | |
| `down_s` | `[62, 256, 1, 4, 4096]` | f32 | |
| NVFP4 (format v2) `gate_up_fp4` | `[62, 256, 512, 1024]` | int32 | eight e2m1 codes per word along K (row `8k' + j` in bits `4j .. 4j + 3`) |
| `gate_up_bs` | `[62, 256, 8, 32, 1024]` | float8_e4m3fn | block-16 scales, K-chunk major |
| `down_fp4` | `[62, 256, 64, 4096]` | int32 | |
| `down_bs` | `[62, 256, 1, 32, 4096]` | float8_e4m3fn | |
| `expert_gs` | `[62, 256, 8, 128]` | f32 | row 0 gate, row 1 up, row 2 down global scale, replicated over lanes |

Bytes per rank (`layout.rank_shapes`): dense 4.60 GiB (76 MiB per layer) in bf16
or 2.55 GiB (42 MiB per layer: 34 MiB of int8 + the 8 MiB bf16 router pair) in
int8, int4 experts 49.4 GiB (3.19 MiB per expert), NVFP4 experts 52.4 GiB (3.38
MiB per expert), embedding + `lm_head` 0.78 GiB (0.59 GiB with the int8
`lm_head`); totals 54.8 GiB (int4) and 57.8 GiB (NVFP4) of the 94.7 GB HBM with
bf16 dense. The KV cache costs 31 KiB per token per rank.

KV cache (`load.zero_caches`): `k_cache` and `v_cache` `[62, B, context, 128]`
bf16 per rank, lanes `[h * 64, (h + 1) * 64)` = local KV head `h`; K is
post-QK-norm and post-RoPE, V raw; the token at absolute position `p` lives in
slot `p` of its row for every layer (no ring buffer for the sliding layers yet).
`context` is a multiple of 256 (the attention tile). The caches are donated and
updated in place (`input_output_aliases`).

## 4. Container formats (`musespark/load.py`)

A container is a directory `<dir>/layout.json` + `<dir>/rank{r}/<name>.bin`
(raw little-endian C-order arrays; layer-stacked arrays keep the layer axis
outermost, so layer `l` of a family is one contiguous byte range) +
`progress.json` (conversion is resumable and idempotent). `layout.json` records
the format, the `Config`, `tp`, the group size, the checkpoint revision and, per
family, the per-rank shape/dtype and the on-disk shape/dtype/byte count.

- **v1 (`musespark-presharded-v1`, `expert_format` int4)**, written by
  `convert_presharded` (`scripts/convert_musespark.sh`) from the bf16 checkpoint:
  int4 values are stored two per byte along the LAST axis, low nibble first
  (`quant.pack_int4`); scales are f32 holding bf16-representable values. The
  converter streams the checkpoint layer by layer (large parallel `preadv` reads
  from NFS, quantization on a process pool). 471 GB on disk.
- **v2 (`musespark-presharded-v2`, `expert_format` nvfp4)**, written by
  `convert_presharded_nvfp4` (`scripts/convert_musespark_nvfp4.sh`) from the
  vendor's NVFP4 repository: the same dense/vector/global families; the experts
  of layers 1..60 are the vendor's bytes re-laid out per rank with pure integer
  transposes (`quant.nvfp4_rows_to_packed`, no float maths), block scales stored
  as their uint8 bits, `expert_gs` from the three `weight_scale_2` tensors (the
  gate and up halves keep their own global scale, which vLLM does not). Layers 0
  and 61, which the vendor left in bf16, are quantized to the same format by
  `quant.quantize_nvfp4_np` (modelopt formula: `ws2 = amax / (6 * 448)`, e4m3
  block scales clamped to `[2^-9, 448]`, RNE) so the kernel has one expert format
  and no per-layer predicate. Shards are downloaded one at a time, converted and
  deleted. 496 GB on disk.

- **int8 dense projections (`dense_format` int8)**, added IN PLACE to a complete
  v1 or v2 container by `python -m musespark.load quantize-dense --dir <dir>`
  (`load.quantize_dense`, ~25 s for the real container on a 32-process pool:
  every rank's bf16 layer slab is memmapped and quantized per output column,
  `o` over the concatenated rows of all ranks, the lm_head per rank). It writes
  the `_i8` / `_s` files next to the untouched bf16 families (+19 GB), records
  the progress under `progress.json["dense_int8"]` (resumable) and, when
  complete, appends the new entries to `layout.json` with `dense_formats:
  [bf16, int8]` and `dense_format: int8` (the preferred format;
  `--no-publish` leaves `layout.json` on bf16 until a later run publishes).
  `load_presharded(..., dense_format=None | "bf16" | "int8")` loads one set of
  dense arrays (`weight_array_names`), default the preferred one;
  `layout.dense_format_of(weights)` lets the kernel, the prefill and the
  reference dispatch on the family names. Quantization error: 1.0-1.3 % relative
  RMS per matrix (bf16 rounding alone is 0.2 %).

`load_presharded` places each rank's arrays on its own device
(`jax.device_put` + `make_array_from_single_device_arrays`) and unpacks the int4
nibbles on the device, so host RAM sees one layer of packed bytes per rank at a
time; `container_expert_format(dir)` tells the callers which format they got and
`layout.expert_format_of(weights)` lets the kernel dispatch on the family names.

## 5. One decode step (`musespark/decode_megakernel.py`)

`make_decode(mesh, cfg, context, batch, greedy=True, return_logits=False,
options=...)` returns a jitted `decode(weights, caches, tokens [B], pos [B]) ->
(next_tokens [B], logits [B, V] | None, caches)` for `B` in {1, 2, 4, 8}; row `b`
is an independent sequence at position `pos[b]`. `jax.jit(shard_map(local))`
strips the rank axis and runs:

1. XLA glue: vocabulary-sharded embedding lookup + `psum`, the parameter-free
   embedding norm `s0 = r16(rms(row))`, the RoPE table of the step.
2. The kernel, one grid-less `pallas_call` with `collective_id`, over all layers,
   the final norm and the `lm_head`, writing the raw logits shard `[8, Vp]` f32
   and the caches.
3. Greedy tokens in-kernel (per-rank masked `(max, argmax)`, one all-gather of
   the eight pairs, lowest id among ties); with `return_logits` (or sampling) the
   glue also gathers the soft-capped, masked full logits.

Per layer (`l` is a `lax.fori_loop` index; the layer kind is arithmetic, no
`lax.cond`):

```
q, kv, g = r16(gemv(x_attn, q[l] / kv[l] / gate[l]))          ring: q, kv, gate
o = attention_layer(...)                                      KV tiles prefetched at layer start
attn_out = r16(all_reduce(gemv(o, o[l])))                     ring: o;    collective 4l
s = alpha * s + beta * r16(rms(attn_out, post_eps)); x_ffn = r16(rms(s, ffn_norm[l]))
h0 = gemv(x_ffn, pre[l]); logits = gemv(x_ffn, router_hi[l]) + gemv(x_ffn, router_lo[l])
route (rank-based top-k); start the first wave of expert DMAs
h1 = r16(rms(all_gather(r16(h0)), pre_expert_norm[l]))       collective 4l + 1
expert_stream -> y_out [K*B, Hm]; Y = all_reduce(y_out)       collective 4l + 2 (hierarchical at B = 8)
m = finalize(Y, w, post_expert_norm[l])
ffn_out = all_gather(r16(gemv(m, post[l])))                   ring: post; collective 4l + 3
t = r16(ffn_out * post_ffn_norm[l]); s = alpha * s + beta * r16(t * rsqrt(mean(t^2) + eps))
x_attn = r16(rms(s, attn_norm[l + 1]))
```

- **Dense ring** (`stream.py`): 12 banks of `[1024, 1024]` bf16 (2 MiB), or
  `[1024, 2048]` int8 (the same 2 MiB) for an int8 container; the static
  schedule (`layout.tile_schedule`) streams q, kv, gate, o, pre, router_hi,
  router_lo, post of every layer and then the `lm_head`; narrow families are
  packed side by side so every load is a full 2 MiB (38 loads per layer in bf16,
  21 in int8). `gemv` re-issues load `g + 12` right after consuming `g`, so 24 MiB
  stay in flight across layer boundaries. Refills issued during the pre/router
  gemvs are held back until the first expert slabs are queued (`defer=next`,
  default at B <= 4), because the DMA engine shares bandwidth among outstanding
  descriptors. Every packed load is consumed by ONE MXU op: the `pack` K-slices
  of x are stacked as row blocks (`[pack * 8, 1024]`, built once per gemv) and
  multiplied by the whole bank; the wanted partial products are the diagonal
  `[8, bn]` blocks of the result (the MXU is weight-push bound, the off-diagonal
  rows are free). In int8 the tiles feed `jnp.dot(bf16, int8)` directly (pushed at
  the bf16 rate, 20 ns per 256 x 256 tile, never `.astype`) and the f32 column
  sums of each N-block are multiplied by the per-column scales after the K sweep
  (`gemv(..., scale=)`); the bf16 router tiles live in the int8 bank through its
  `.bitcast(bf16)` view (`[512, 2048]`). Measured on chip (16 real-width layers,
  `tests/test_musespark_stream.py`): bf16 26.4 us per layer (3.0 TB/s, DMA-bound),
  int8 17.5 us (DMA-only floor 16.1; one narrow dot per K-tile gave 21.1 --
  the issue latency and int8 conversion of 72 small dots per layer), `lm_head`
  130 -> 68 us.
- **Attention** (`attention.py`): 256-token KV tiles, three buffers per cache
  (two in flight, one consumed), block range `[lo // 256, pos // 256]` with `lo =
  max(0, pos - 2047)` on sliding layers and 0 on full layers; a block-diagonal
  `Q2 [16, 128]` scores both local KV heads in one MXU op per tile; the current
  token is patched into the resident tile and the last tile is written back to
  HBM. The tile DMAs are issued at the start of the layer (E4 in `perf_log.md`:
  -4.5 us per layer).
- **Routing** (`moe.route_from_logits`): `sc = sigmoid(logits * 2^-2.5)`, top-k
  on `sc + bias` by one `[E, E]` rank comparison per row (bit-identical to
  iterative argmax with ties to the lowest index), weights `sc[idx] / (sum +
  1e-15)`; results are lane-dense `[B, 128]` tiles.
- **Expert stream** (`moe.expert_stream`, int4; `fp4.expert_stream`, NVFP4):
  every distinct expert of the batch is processed once in ascending id, in waves
  of four slots whose DMAs are issued as early as possible; at B = 1 the waves are
  static. int4 slabs travel as packed int8 bytes (`ref.bitcast(int8)`, 3.1 MiB per
  expert in VMEM) and are viewed back as int4 for `jnp.dot(bf16, int4)`; group
  scales are applied by expanding the input to a block-diagonal `[K/KC, GPC*8, KC]`
  LHS so each group's partial product lands in its own row block and the scale is
  a VPU multiply. NVFP4 slabs are bitcast to `float4_e2m1fn` and converted for
  free to `float8_e4m3fn`; the block-16 e4m3 scales use the same block-diagonal
  idiom with 32 x B rows per 512-chunk, and the three per-expert global scales
  are scalar multiplies. `y_out[k*B + b]` holds the rank's partial down output of
  route slot `k` of row `b`; `finalize` applies `post_expert_norm` on the reduced
  sum and the routing weights.
- **Per-layer vectors** are double-buffered by `l % 2` and prefetched one layer
  ahead; the `lm_head` accumulates into an `[8, Vp]` f32 VMEM output.

Options (`frozenset` of strings): `interpret` (CPU tests), `aux_hidden` (return
the residual stream after every layer), `moe_slots=N`, `banks=N`,
`hier=on|off`, `wire=bf16|f32`, `flush=`, `kv_late`, `defer=`, and the
profiling-only `skip=...`; see the module docstring.

## 6. VMEM budget (`decode_megakernel.vmem_budget`)

`vmem_limit_bytes = 64 MiB`; explicit allocations are kept <= 58 MiB
(`VMEM_EXPLICIT_LIMIT`) because the compiler adds ~2.7 MiB of spill slots at
real widths. `default_geometry` picks the deepest ring in (12, 10, 8) that still
fits four expert slots; at real widths that is always 12 banks and 4 slots:

| config | total | ring | expert slots | collectives | attention | vectors | logits acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| int4, B = 1 | 41.80 MiB | 24.00 | 14.01 | 2.00 | 0.53 | 0.38 | 0.78 |
| int4, B = 8 | 50.86 MiB | 24.00 | 14.89 | 5.75 | 4.22 | 0.38 | 0.78 |
| NVFP4, B = 1 | 41.69 MiB | 24.00 | 13.90 | 2.00 | 0.53 | 0.38 | 0.78 |
| NVFP4, B = 8 | 52.50 MiB | 24.00 | 16.53 | 5.75 | 4.22 | 0.38 | 0.78 |

Slot count sweep (`perf_log.md` E9, L = 8 layers, real widths): 3 / 4 / 5 / 6 / 8
slots = 668 / 653 / 685 / 690 / 824 us at B = 1; four slots win because the DMA
engine round-robins among outstanding descriptors, so more slabs in flight all
land later. Unsigned integer vector ops crash the core, so nibble unpacking
happens on the host or in XLA, never in the kernel.

## 7. Collectives (`musespark/collectives.py`)

Symmetric collectives over the eight cores addressed as `rank ^ offset`
(`pl.DeviceIdType.MESH`); each rank sends to the seven others via
`make_async_remote_copy` and waits for its seven sends and seven receives, so DMA
semaphores are the only synchronisation. The kernel calls `barrier()` once at
entry. Every payload column block is summed by exactly one rank in slot order
0..7 and then gathered, so every rank receives the bit-identical result.

- `all_reduce_rows(x [R, W])`: reduce-scatter over `W / 8` column blocks +
  all-gather (14 messages, 2 latency phases). `wire=bf16` (default) rounds each
  rank's partial to bf16 before the fixed-order f32 sum and ships the reduced
  blocks as bf16; the result is `r16`'d by the caller anyway. Measured on the
  `[8, 8192]` o-proj payload: 9.4 us f32 vs 7.3 us bf16 wire; on the `[64, 4096]`
  B = 8 expert payload 25.4 vs 14.5 us (`perf_log.md` E1/E3).
- `all_reduce_rows(..., hierarchical=True)`: pair reduce-scatter of the column
  halves over the sibling core, reduce-scatter of the half's quarters across the
  four chips (one message per link), all-gather of the quarters, pair
  all-gather. Cross-chip bytes halve at the price of four latency phases; it
  wins only for the large payload (`[64, 4096]`: 13.5 vs 14.5 us bf16 wire,
  19.9 vs 25.4 us f32), hence `hier` defaults to on at B >= 8 and off below
  (`[8, 8192]`: 9.3 vs 7.3 us). In the kernel: -1.2 us per layer at B = 8
  (E13).
- `all_gather_rows(x [R, w])`: direct, rank `r`'s shard lands in columns
  `r*w:(r+1)*w`; direct beats hierarchical for every gather payload measured.
- Buffers and semaphores are double-buffered by the parity of a per-rank
  collective counter (four collectives per layer: `4l .. 4l + 3`), which is
  sufficient because a rank can only start collective `c + 2` after every peer
  has finished consuming the buffers of `c`.

At B = 1 every collective is latency-bound (~3.5-4.7 us per phase); the four
collectives cost ~14 us of the ~55 us per layer.

## 8. Numerics and rounding policy

The reference model (`musespark/__init__.py`) follows the model spec's rounding
points exactly and the kernel and the prefill reproduce them:

- f32 residual stream; `r16` (`lax.reduce_precision` to bf16 and back, honoured
  even under XLA's excess-precision default) at every norm output and every
  branch output; bf16 x bf16 GEMMs with f32 accumulation; `rms(x) = x *
  rsqrt(mean(x^2) + eps)` with `rms_eps = 1e-5` for the pre-norms, the QK norms
  and the attention-output norm and `post_eps = 1e-8` for the post-norms, whose
  effective weights are precomputed as `r16(1 + gamma)` and, for
  `post_ffn_norm` / `post_expert_norm`, multiply before the norm;
- the f32 router as `dot(x, hi) + dot(x, lo)`; the residual gates as
  `alpha * s + beta * branch` with `alpha = sqrt(max(sigmoid(-u) * (1 +
  sigmoid(u)), 1e-3))`, `beta = sigmoid(u)`, `u = gate / 0.3`;
- attention scores from bf16 q/k products accumulated in f32, f32 softmax and
  `P . V` (the kernel's hi/lo MXU mode), `o = r16(rms(o) * sigmoid(g))`;
- experts: `gu = x @ W` in f32, `a = r16(silu(gate) * up)`, `y` in f32, the
  post-expert norm on the cross-rank sum, `m = r16(sum_k w_k * yn_k)`;
- int4 g128: `scale = bf16(absmax / 7)`, `q = clip(round_half_even(w / scale),
  -8, 7)`, so `dequant = q * scale` is exact in f32 (and bf16); the kernel's
  `bf16 x int4` dot with block-diagonal scales equals the dequantized bf16 dot up
  to f32 summation order (rel. err ~1e-6 measured);
- NVFP4: `w = e2m1 * e4m3 * gs`; `e2m1 * e4m3` has <= 6 significant bits and is
  exact in bf16, only the f32 global-scale multiply rounds; measured 9.5 % rel.
  RMS vs the bf16 weights for both the vendor layers and our RTN layers 0/61
  (int4 g128: 11.7 %), `nvfp4_container_v2.md`;
- the bf16 wire of the two per-layer all-reduces rounds each partial to bf16
  before the f32 sum (real-weights logits max |diff| vs the XLA oracle 0.497
  instead of 0.470 with f32 partials, argmax and top-5 unchanged, E3);
- run TPU programs with `XLA_FLAGS=--xla_allow_excess_precision=false` so the
  XLA glue's `r16` matches the reference bit for bit (the launchers set it).

Tolerances: MINI decode vs reference max |soft-capped logit diff| <= 5e-2 with
greedy-token agreement; residual streams <= 0.1 (unit RMS); cache slots <= 0.1.
Real weights (`tests/test_musespark_real_decode.py`): logits <= 1.0 with the
oracle's argmax, residual stream <= 10 % relative RMS (measured 0.47 / 4.5 %),
layer-0 K-cache slots <= 0.1 vs the prefill. The prefill test asserts the
layer-0 K/V slots bit-exact against the reference on CPU. On TPU the f32 result
of a bf16 GEMM depends on its N-shape: the per-rank `[H, 2 * kvh * D]` and the
reference's `[H, nKV * D]` K projections differ by ~1e-7 in most elements
(measured on MINI: 13k of 19k, max 6e-7), which flips the bf16 rounding of a rare
projection value (5 of 18944 slots) and RoPE carries the flip to both halves of
its pair; the test therefore checks every layer-0 slot within two bf16 ulps of
its RoPE-pair magnitude and >= 99.5 % bit-exact slots there (a layout / RoPE /
mask bug moves most slots by O(0.3)).

## 9. Prefill (`musespark/prefill.py`)

`make_prefill(mesh, cfg, context, tp)` returns a jitted `prefill(weights, caches,
tokens [Tp], length, row) -> (logits_last [V], caches)`: one `jax.shard_map`
over the kernel's per-rank weights, the prompt right-padded to a 64-token bucket
(one executable per bucket), pad rows masked everywhere. Attention is blocked in
512-query blocks with f32 scores; experts use `lax.ragged_dot` on the `Tp * 8`
routes sorted by expert, on the layer's experts dequantized to bf16 inside the
layer loop (int4: `quant.dequantize_int4`; NVFP4: `fp4.dequant_fp4_pallas` on
TPU, XLA elsewhere, bit-identical); communication per layer is a `psum` of the
o-proj partials, all-gathers of the `pre`, expert-mixture and `post` shards and a
`psum_scatter` of the expert partials. The K/V of positions `0 .. Tp - 1` are
written to cache row `row` in the kernel layout; decode continues at
`pos = length`. Measured per prompt (B = 1, `validate_musespark_decode` logs of
section 10): 421 / 462 / 475 / 522 / 657 ms for buckets 64 / 192 / 512 / 1024 /
2048 on int4, 272 / 318 / 330 / 375 / 507 ms on NVFP4 (the Pallas dequant made
the fp4 prefill faster than the int4 XLA dequant).

## 10. Performance summary

Official benchmark (`scripts/validate_musespark_decode.py --tasks bench`,
context 4096, every row prompt 0 of 151 tokens, 128 timed steps of 16 per call):

| B | int4 ms/step (`logs/validate_musespark_decode_20260926T052025Z.log`) | tok/s aggr. | HBM floor | NVFP4 ms/step (`..._20260926T051551Z.log`) | tok/s aggr. |
|---|---:|---:|---:|---:|---:|
| 1 | 3.592 | 278 | 2.19 ms (61 %) | 3.820 | 262 |
| 2 | 3.705 | 540 | 2.71 ms | 6.975 | 287 |
| 4 | 3.912 | 1022 | 3.75 ms | 6.598 | 606 |
| 8 | 4.291 | 1864 | 5.82 ms | 10.398 | 769 |

With the int8 dense projections + `lm_head` (`dense_format` int8, the
container's default since `quantize-dense` ran; `perf_log.md` E15), same
benchmark, NVFP4 experts:

| B | bf16 dense ms/step (`logs/..._20260926T070309Z.log`) | int8 dense ms/step (default path, `logs/..._20260926T083121Z.log`) | tok/s per row | tok/s aggr. |
|---|---:|---:|---:|---:|
| 1 | 3.565 | **3.260** | **306.8** | 307 |
| 2 | 4.070 | 3.657 | 273.4 | 547 |
| 4 | 4.626 | 4.245 | 235.5 | 942 |
| 8 | 4.861 | 4.462 | 224.1 | 1793 |

(int8 halves the dense bytes: 42 MiB per layer instead of 76, `lm_head` 200 MiB
instead of 400; the layer's dense phase is then MXU-bound at the bf16 push rate,
17.5 us per layer in the stream benchmark vs 26.4 us for bf16.)

The first kernel version (`logs/validate_musespark_decode_20260926T022450Z.log`)
took 4.531 / 4.712 / 5.096 / 6.012 ms; the gains came from the bf16 wire, KV
prefetch, deferred ring refills, packed 2 MiB bank loads, packed int4 slots,
rank-based top-k with static B = 1 waves, in-kernel greedy and the hierarchical
B = 8 all-reduce (`perf_log.md` E3-E13). Note the bench rows are identical, so
even at B = 8 only eight distinct experts are loaded per layer; the "floor"
column is the harness's `8 * B` experts per row bound.

Where the B = 1 step goes (`perf_log.md`, phase_bench at real widths): ~55 us per
layer + ~195 us fixed. Per layer: dense ring 25 us (76 MiB at ~3 TB/s, the
bytes), collectives ~14 us (four latency-bound phases), expert phase ~9 us (8 us
of bytes), attention 1.5 us, routing 0.7 us. Fixed: `lm_head` 126 us (400 MiB)
plus ~70 us of dispatch, XLA glue and the barrier.

## 11. Tests and validation

- CPU, eight virtual devices (`JAX_PLATFORMS=cpu
  XLA_FLAGS=--xla_force_host_platform_device_count=8`), Pallas bodies in
  interpret mode: `tests/test_musespark_reference.py` (spec re-implementation,
  rope equivalence, routing ties), `_quant.py`, `_layout.py`, `_load.py`
  (container round trips), `_tokenizer.py`, `_sampling.py`, `_attention.py`,
  `_moe.py`, `_collectives.py`, `_stream.py`, `_fp4.py`, `_decode.py`,
  `_decode_fp4.py`, `_prefill.py`: 177 passed, 26 skipped (incl. the int8 dense
  container round trips, the int8 / packed ring and the int8 MINI decode).
- TPU: the same files on hardware (single chip for the component tests, all
  eight cores for collectives / decode / prefill, plus the real-width smoke test
  of `_decode.py` that prints the VMEM budget and per-layer timings);
  `tests/test_musespark_real_prefill.py` and `_real_decode.py` on the real
  container against the oracle of `scripts/validate_musespark_prefill.py`
  (`--dense-format` selects the oracle's dense projections;
  `MUSESPARK_DENSE_FORMAT` the test's). On the NVFP4 container the prompt-0
  replay sits on a routing near-tie: kernel vs XLA prefill is 1.6 (bf16, before
  the int8 work) to 9.8 (int8) in max |logit diff| there, 0.7-1.1 on prompts 1/2,
  with the same argmax and teacher-forced hit rate (`perf_log.md` E15).
- `scripts/validate_musespark_decode.py` (replay / generate / bench on real
  weights), `scripts/compare_official_musespark.py` (JAX reference vs the
  official sglang layer maths, CPU, real bf16 weights), `scripts/eval_musespark_gsm8k.sh`
  (lm-evaluation-harness through the server).

## 12. Known open items

- **NVFP4 at B > 1.** `fp4.expert_stream` is MXU-bound: the block-diagonal LHS
  grows with `32 * B` rows (0.87 / 1.87 / 1.67 / 2.60 us per gate_up slice at B =
  1 / 2 / 4 / 8) and the stream still carries four wave-body variants, giving
  7.0 / 6.6 / 10.4 ms at B = 2 / 4 / 8 vs 3.7 / 3.9 / 4.3 ms for int4. Candidates
  (`nvfp4_feasibility.md` section 3): split the experts between the MXU
  block-diagonal path and a VPU dequant path (~17-20 us per layer at B = 8
  projected), fewer wave-body variants.
- **In-kernel embedding** (the XLA lookup + `psum` glue is ~20 us per step).
- **Sliding-window ring buffer**: the caches are contiguous over the full
  context for every layer; the 46 sliding layers only ever read the last 2048
  slots, so a 2048-slot ring per sliding layer would cut the KV memory by ~2/3
  and simplify long contexts.
- Smaller items from `perf_log.md`: fuse the `pre` all-gather into the router
  phase (-4 us per layer), overlap the o-proj all-reduce with the pre/router
  gemvs (-5 us per layer, needs the post-attention boundary restructured).
- The prefill is XLA-only (no fused prefill kernel) and the server answers one
  request at a time on row 0.
