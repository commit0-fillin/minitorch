from __future__ import annotations
import random
from typing import Iterable, Optional, Sequence, Tuple, Union
import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias
from .operators import prod
MAX_DIMS = 32

class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""
    pass
Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]
UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]

def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """
    return sum(i * s for i, s in zip(index, strides))

def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = ordinal % shape[i]
        ordinal //= shape[i]

def broadcast_index(big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    offset = len(big_shape) - len(shape)
    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[offset + i]
        else:
            out_index[i] = 0

def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    max_len = max(len(shape1), len(shape2))
    new_shape = []
    for i in range(max_len):
        dim1 = shape1[len(shape1) - 1 - i] if i < len(shape1) else 1
        dim2 = shape2[len(shape2) - 1 - i] if i < len(shape2) else 1
        if dim1 == 1 or dim2 == 1 or dim1 == dim2:
            new_shape.append(max(dim1, dim2))
        else:
            raise IndexingError(f"Cannot broadcast shapes {shape1} and {shape2}")
    return tuple(reversed(new_shape))

class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(self, storage: Union[Sequence[float], Storage], shape: UserShape, strides: Optional[UserStrides]=None):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)
        if strides is None:
            strides = strides_from_shape(shape)
        assert isinstance(strides, tuple), 'Strides must be tuple'
        assert isinstance(shape, tuple), 'Shape must be tuple'
        if len(strides) != len(shape):
            raise IndexingError(f'Len of strides {strides} must match {shape}.')
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last_stride = 1
        for s, stride in zip(reversed(self.shape), reversed(self.strides)):
            if stride != last_stride:
                return False
            last_stride *= s
        return True

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        if len(order) != len(self.shape):
            raise IndexingError(f"Cannot permute {len(self.shape)} dimensions with {len(order)} indices")
        
        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)
        return TensorData(self._storage, new_shape, new_strides)
