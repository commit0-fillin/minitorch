from typing import Callable, Optional
import numba
from numba import cuda
from .tensor import Tensor
from .tensor_data import MAX_DIMS, Shape, Storage, Strides, TensorData, broadcast_index, index_to_position, shape_broadcast, to_index
from .tensor_ops import MapProto, TensorOps
to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)
THREADS_PER_BLOCK = 32

class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        def _map(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = x.zeros(x.shape)
            
            # Launch CUDA kernel
            threads_per_block = THREADS_PER_BLOCK
            blocks_per_grid = (x.size + threads_per_block - 1) // threads_per_block
            cuda_kernel = cuda.jit(device=True)(fn)
            tensor_map_kernel[blocks_per_grid, threads_per_block](
                x._tensor._storage, x._tensor.shape, x._tensor.strides,
                out._tensor._storage, out._tensor.shape, out._tensor.strides,
                cuda_kernel
            )
            return out
        return _map

def tensor_map(fn: Callable[[float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """
    def _map(in_storage: Storage, in_shape: Shape, in_strides: Strides,
             out_storage: Storage, out_shape: Shape, out_strides: Strides) -> None:
        i = cuda.grid(1)
        if i < out_storage.size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            in_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_position = index_to_position(in_index, in_strides)
            out_position = index_to_position(out_index, out_strides)
            out_storage[out_position] = fn(in_storage[in_position])

    return cuda.jit()(_map)

def tensor_zip(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.

    Returns:
        Tensor zip function.
    """
    def _zip(a_storage: Storage, a_shape: Shape, a_strides: Strides,
             b_storage: Storage, b_shape: Shape, b_strides: Strides,
             out_storage: Storage, out_shape: Shape, out_strides: Strides) -> None:
        i = cuda.grid(1)
        if i < out_storage.size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            a_index = cuda.local.array(MAX_DIMS, numba.int32)
            b_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_position = index_to_position(a_index, a_strides)
            b_position = index_to_position(b_index, b_strides)
            out_position = index_to_position(out_index, out_strides)
            out_storage[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return cuda.jit()(_zip)

def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // 	ext{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    shared = cuda.shared.array(THREADS_PER_BLOCK, numba.float64)
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    
    i = bid * THREADS_PER_BLOCK + tid
    
    if i < size:
        shared[tid] = a[i]
    else:
        shared[tid] = 0.0
    
    cuda.syncthreads()
    
    s = THREADS_PER_BLOCK // 2
    while s > 0:
        if tid < s and i + s < size:
            shared[tid] += shared[tid + s]
        cuda.syncthreads()
        s //= 2
    
    if tid == 0:
        out[bid] = shared[0]
jit_sum_practice = cuda.jit()(_sum_practice)

def tensor_reduce(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.

    Returns:
        Tensor reduce function.

    """
    def _reduce(out: Storage, out_shape: Shape, out_strides: Strides,
                a: Storage, a_shape: Shape, a_strides: Strides,
                reduce_dim: int) -> None:
        
        shared = cuda.shared.array(THREADS_PER_BLOCK, numba.float64)
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        
        a_size = int(prod(a_shape))
        out_size = int(prod(out_shape))
        
        reduce_size = a_shape[reduce_dim]
        
        for i in range(bid, out_size, cuda.gridDim.x):
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            
            # Initialize with the first element
            a_index = out_index.copy()
            a_index[reduce_dim] = 0
            shared[tid] = a[index_to_position(a_index, a_strides)]
            
            # Reduce within a block
            for j in range(1, reduce_size):
                a_index[reduce_dim] = j
                val = a[index_to_position(a_index, a_strides)]
                shared[tid] = fn(shared[tid], val)
            
            # Reduce within shared memory
            cuda.syncthreads()
            s = THREADS_PER_BLOCK // 2
            while s > 0:
                if tid < s:
                    shared[tid] = fn(shared[tid], shared[tid + s])
                cuda.syncthreads()
                s //= 2
            
            # Write result to global memory
            if tid == 0:
                out[index_to_position(out_index, out_strides)] = shared[0]
    
    return cuda.jit()(_reduce)

def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square
    """
    BLOCK_DIM = 32
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    row = cuda.blockIdx.y * BLOCK_DIM + ty
    col = cuda.blockIdx.x * BLOCK_DIM + tx
    
    if row < size and col < size:
        tmp = 0.0
        for k in range(0, size, BLOCK_DIM):
            if k + tx < size:
                shared_a[ty, tx] = a[row * size + k + tx]
            if k + ty < size:
                shared_b[ty, tx] = b[(k + ty) * size + col]
            
            cuda.syncthreads()
            
            for i in range(min(BLOCK_DIM, size - k)):
                tmp += shared_a[ty, i] * shared_b[i, tx]
            
            cuda.syncthreads()
        
        out[row * size + col] = tmp
jit_mm_practice = cuda.jit()(_mm_practice)

def _tensor_matrix_multiply(out: Storage, out_shape: Shape, out_strides: Strides, out_size: int, a_storage: Storage, a_shape: Shape, a_strides: Strides, b_storage: Storage, b_shape: Shape, b_strides: Strides) -> None:
    """
    CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    BLOCK_DIM = 32
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    
    bx, by, bz = cuda.gridDim.x, cuda.gridDim.y, cuda.gridDim.z
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    
    batch = bz * tz + by
    row = bx * BLOCK_DIM + tx
    col = by * BLOCK_DIM + ty
    
    if batch < out_shape[0] and row < out_shape[1] and col < out_shape[2]:
        tmp = 0.0
        for k in range(0, a_shape[-1], BLOCK_DIM):
            if k + ty < a_shape[-1] and row < a_shape[-2]:
                shared_a[tx, ty] = a_storage[index_to_position((batch, row, k + ty), a_strides)]
            else:
                shared_a[tx, ty] = 0.0
            
            if k + tx < b_shape[-2] and col < b_shape[-1]:
                shared_b[tx, ty] = b_storage[index_to_position((batch, k + tx, col), b_strides)]
            else:
                shared_b[tx, ty] = 0.0
            
            cuda.syncthreads()
            
            for i in range(BLOCK_DIM):
                tmp += shared_a[tx, i] * shared_b[i, ty]
            
            cuda.syncthreads()
        
        out[index_to_position((batch, row, col), out_strides)] = tmp
tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
