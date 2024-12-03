from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numba import njit, prange
from .tensor_data import MAX_DIMS, broadcast_index, index_to_position, shape_broadcast, to_index
from .tensor_ops import MapProto, TensorOps
if TYPE_CHECKING:
    from typing import Callable, Optional
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides
to_index = njit(inline='always')(to_index)
index_to_position = njit(inline='always')(index_to_position)
broadcast_index = njit(inline='always')(broadcast_index)

class FastOps(TensorOps):

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        def _map(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            
            tensor_map(fn)(
                a._tensor._storage,
                a._tensor._shape,
                a._tensor._strides,
                out._tensor._storage,
                out._tensor._shape,
                out._tensor._strides,
            )
            return out
        
        return _map

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        def _zip(a: Tensor, b: Tensor) -> Tensor:
            out_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(out_shape)
            
            tensor_zip(fn)(
                a._tensor._storage,
                a._tensor._shape,
                a._tensor._strides,
                b._tensor._storage,
                b._tensor._shape,
                b._tensor._strides,
                out._tensor._storage,
                out._tensor._shape,
                out._tensor._strides,
            )
            return out
        
        return _zip

    @staticmethod
    def reduce(fn: Callable[[float, float], float], start: float=0.0) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        def _reduce(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            
            tensor_reduce(fn)(
                a._tensor._storage,
                a._tensor._shape,
                a._tensor._strides,
                out._tensor._storage,
                out._tensor._shape,
                out._tensor._strides,
                dim,
            )
            return out
        
        return _reduce

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """
        assert a.shape[-1] == b.shape[-2], "Incompatible matrix dimensions for multiplication"
        
        # Determine output shape
        out_shape = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        out_shape.extend([a.shape[-2], b.shape[-1]])
        
        # Create output tensor
        out = a.zeros(tuple(out_shape))
        
        # Perform matrix multiplication
        tensor_matrix_multiply(
            out._tensor._storage,
            out._tensor._shape,
            out._tensor._strides,
            a._tensor._storage,
            a._tensor._shape,
            a._tensor._strides,
            b._tensor._storage,
            b._tensor._shape,
            b._tensor._strides,
        )
        
        return out

@njit(parallel=True)
def tensor_map(fn: Callable[[float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """
    def _map(in_storage: Storage, in_shape: Shape, in_strides: Strides,
             out_storage: Storage, out_shape: Shape, out_strides: Strides) -> None:
        out_size = int(np.prod(out_shape))
        in_size = int(np.prod(in_shape))
        
        # Check if tensors are stride-aligned
        is_stride_aligned = np.array_equal(in_strides, out_strides) and np.array_equal(in_shape, out_shape)
        
        if is_stride_aligned:
            for i in prange(out_size):
                out_storage[i] = fn(in_storage[i])
        else:
            for i in prange(out_size):
                out_index = np.empty(len(out_shape), dtype=np.int32)
                in_index = np.empty(len(in_shape), dtype=np.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                in_position = index_to_position(in_index, in_strides)
                out_position = index_to_position(out_index, out_strides)
                out_storage[out_position] = fn(in_storage[in_position])
    
    return _map

@njit(parallel=True)
def tensor_zip(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """
    def _zip(a_storage: Storage, a_shape: Shape, a_strides: Strides,
             b_storage: Storage, b_shape: Shape, b_strides: Strides,
             out_storage: Storage, out_shape: Shape, out_strides: Strides) -> None:
        out_size = int(np.prod(out_shape))
        
        # Check if tensors are stride-aligned
        is_stride_aligned = (np.array_equal(a_strides, out_strides) and
                             np.array_equal(b_strides, out_strides) and
                             np.array_equal(a_shape, out_shape) and
                             np.array_equal(b_shape, out_shape))
        
        if is_stride_aligned:
            for i in prange(out_size):
                out_storage[i] = fn(a_storage[i], b_storage[i])
        else:
            for i in prange(out_size):
                out_index = np.empty(len(out_shape), dtype=np.int32)
                a_index = np.empty(len(a_shape), dtype=np.int32)
                b_index = np.empty(len(b_shape), dtype=np.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                a_position = index_to_position(a_index, a_strides)
                b_position = index_to_position(b_index, b_strides)
                out_position = index_to_position(out_index, out_strides)
                out_storage[out_position] = fn(a_storage[a_position], b_storage[b_position])
    
    return _zip

@njit(parallel=True)
def tensor_reduce(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """
    def _reduce(a_storage: Storage, a_shape: Shape, a_strides: Strides,
                out_storage: Storage, out_shape: Shape, out_strides: Strides,
                reduce_dim: int) -> None:
        out_size = int(np.prod(out_shape))
        reduce_size = a_shape[reduce_dim]
        
        for i in prange(out_size):
            out_index = np.empty(len(out_shape), dtype=np.int32)
            to_index(i, out_shape, out_index)
            
            # Initialize accumulator
            a_index = out_index.copy()
            a_index[reduce_dim] = 0
            accumulator = a_storage[index_to_position(a_index, a_strides)]
            
            # Reduce over the specified dimension
            for j in range(1, reduce_size):
                a_index[reduce_dim] = j
                a_position = index_to_position(a_index, a_strides)
                accumulator = fn(accumulator, a_storage[a_position])
            
            out_position = index_to_position(out_index, out_strides)
            out_storage[out_position] = accumulator
    
    return _reduce

@njit(parallel=True, fastmath=True)
def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if len(a_shape) == 3 else 0
    b_batch_stride = b_strides[0] if len(b_shape) == 3 else 0
    
    assert a_shape[-1] == b_shape[-2]
    
    for i in prange(out_shape[0]):  # Batch dimension
        for j in range(out_shape[1]):  # Rows of output
            for k in range(out_shape[2]):  # Columns of output
                a_batch_offset = i * a_batch_stride
                b_batch_offset = i * b_batch_stride
                
                acc = 0.0
                for m in range(a_shape[-1]):  # Inner dimension
                    a_val = a_storage[a_batch_offset + j * a_strides[-2] + m * a_strides[-1]]
                    b_val = b_storage[b_batch_offset + m * b_strides[-2] + k * b_strides[-1]]
                    acc += a_val * b_val
                
                out[i * out_strides[0] + j * out_strides[1] + k * out_strides[2]] = acc
tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
