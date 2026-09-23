"""Experimental raw VMEM views; every new view must be overwritten before reading."""
import dataclasses
import math
from jax import tree_util
from jax._src.state import types
from jax._src.pallas.mosaic import lowering
from jax._src.lib import tpu
from jax._src.lib.mlir import ir

@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Reinterpret(types.ReshapeTransform):
    pass

_original = lowering._reshape_memref

def _lower(ref, transform, aval, block_shape):
    if not isinstance(transform, Reinterpret):
        return _original(ref, transform, aval, block_shape)
    transform._validate_shape(tuple(block_shape))
    original_type = ir.MemRefType(ref.type)
    bits = aval.dtype.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    columns = transform.shape[-1] // 128
    strides = [columns, 1]
    minor_dimension = transform.shape[-2]
    for dimension in reversed(transform.shape[:-2]):
        minor_tiles = (minor_dimension + rows - 1) // rows
        strides.insert(0, strides[0] * minor_tiles)
        minor_dimension = dimension
    inner = f"({packing},1)" if packing > 1 else ""
    layout = ir.Attribute.parse(f"#tpu.tiled<({rows},128){inner},[{",".join(map(str, strides))}]>")
    result_type = ir.MemRefType.get(transform.shape, original_type.element_type, layout=layout, memory_space=original_type.memory_space)
    alias = tpu.reinterpret_cast(result_type, ref)
    erased = ir.MemRefType.get(transform.shape, original_type.element_type, memory_space=original_type.memory_space)
    return tpu.erase_memref_layout(erased, alias), transform.shape

lowering._reshape_memref = _lower

def storage_size(shape, dtype):
    bits = dtype.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    row_tiles = (shape[-2] + rows - 1) // rows
    return math.prod(shape[:-2]) * row_tiles * (shape[-1] // 128) * 4096

def view(ref, shape, dtype):
    ref = ref.bitcast(dtype)
    return types.TransformedRef(ref.ref, (*ref.transforms, Reinterpret(tuple(shape))))
