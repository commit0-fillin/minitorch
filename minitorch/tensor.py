"""
Implementation of the core Tensor object for autodifferentiation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData
from .tensor_functions import EQ, LT, Add, All, Copy, Exp, Inv, IsClose, Log, MatMul, Mul, Neg, Permute, ReLU, Sigmoid, Sum, View, tensor
if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union
    import numpy.typing as npt
    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend
    TensorLike = Union[float, int, 'Tensor']

@dataclass
class History:
    """
    `History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """
    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()
_tensor_count = 0

class Tensor:
    """
    Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """
    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(self, v: TensorData, back: Optional[History]=None, name: Optional[str]=None, backend: Optional[TensorBackend]=None):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)
        self.f = backend

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """
        Returns:
             Converted to numpy array
        """
        return self._tensor.to_numpy()

    @property
    def shape(self) -> UserShape:
        """
        Returns:
             shape of the tensor
        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """
        Returns:
             int : size of the tensor
        """
        return self._tensor.size

    @property
    def dims(self) -> int:
        """
        Returns:
             int : dimensionality of the tensor
        """
        return len(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (float, int)):
            return tensor([b], backend=self.backend)
        elif isinstance(b, Tensor):
            return b
        else:
            raise TypeError(f"Can't convert {type(b)} to Tensor")

    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        return self + b

    def __rmul__(self, b: TensorLike) -> Tensor:
        return self * b

    def sum(self, dim: Optional[int]=None) -> Tensor:
        """Compute the sum over dimension `dim`"""
        return Sum.apply(self, dim)

    def mean(self, dim: Optional[int]=None) -> Tensor:
        """Compute the mean over dimension `dim`"""
        sum_result = self.sum(dim)
        if dim is None:
            return sum_result / self.size
        else:
            return sum_result / self.shape[dim]

    def permute(self, *order: int) -> Tensor:
        """Permute tensor dimensions to *order"""
        return Permute.apply(self, order)

    def view(self, *shape: int) -> Tensor:
        """Change the shape of the tensor to a new shape with the same size"""
        return View.apply(self, shape)

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    @staticmethod
    def make(storage: Union[Storage, List[float]], shape: UserShape, strides: Optional[UserStrides]=None, backend: Optional[TensorBackend]=None) -> Tensor:
        """Create a new tensor from data"""
        tensor_data = TensorData(storage, shape, strides)
        return Tensor(tensor_data, backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """
        Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Parameters:
            other : backward tensor (must broadcast with self)

        Returns:
            Expanded version of `other` with the right derivatives

        """
        if self.shape == other.shape:
            return other
        
        expanded_shape = [max(s, o) for s, o in zip(self.shape, other.shape)]
        expanded = other.f.zeros(expanded_shape)
        
        for idx in np.ndindex(expanded_shape):
            other_idx = tuple(i if s > 1 else 0 for i, s in zip(idx, other.shape))
            expanded[idx] = other[other_idx]
        
        return expanded

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x : value to be accumulated
        """
        assert self.is_leaf(), "Derivative can only be accumulated on leaf variables."
        if self.grad is None:
            self.grad = x
        else:
            self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is None or self.history.last_fn is None

    def zero_grad_(self) -> None:
        """
        Reset the derivative on this variable.
        """
        self.grad = None
