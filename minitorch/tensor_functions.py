"""
Implementation of the autodifferentiation Functions for Tensor.
"""
from __future__ import annotations
import random
from typing import TYPE_CHECKING
import numpy as np
import minitorch
from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend
if TYPE_CHECKING:
    from typing import Any, List, Tuple
    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape

def wrap_tuple(x):
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)

class Function:
    pass

class Neg(Function):
    pass

class Inv(Function):
    pass

class Add(Function):
    pass

class Mul(Function):
    pass

class Sigmoid(Function):
    pass

class ReLU(Function):
    pass

class Log(Function):
    pass

class Exp(Function):
    pass

class Sum(Function):
    pass

class All(Function):
    pass

class LT(Function):
    pass

class EQ(Function):
    pass

class IsClose(Function):
    pass

class Permute(Function):
    pass

class View(Function):
    pass

class Copy(Function):
    pass

class MatMul(Function):
    pass

def zeros(shape: UserShape, backend: TensorBackend=SimpleBackend) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    size = int(operators.prod(shape))
    return Tensor.make([0.0] * size, shape, backend=backend)

def rand(shape: UserShape, backend: TensorBackend=SimpleBackend, requires_grad: bool=False) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    size = int(operators.prod(shape))
    data = [random.random() for _ in range(size)]
    tensor = Tensor.make(data, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor

def _tensor(ls: Any, shape: UserShape, backend: TensorBackend=SimpleBackend, requires_grad: bool=False) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    def flatten(lst):
        if isinstance(lst, (float, int)):
            return [float(lst)]
        return [float(item) for sublist in lst for item in (flatten(sublist) if isinstance(sublist, (list, tuple)) else [sublist])]

    flat_data = flatten(ls)
    if int(operators.prod(shape)) != len(flat_data):
        raise ValueError(f"Shape {shape} is not compatible with data of length {len(flat_data)}")

    tensor = Tensor.make(flat_data, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor

def tensor(ls: Any, backend: TensorBackend=SimpleBackend, requires_grad: bool=False) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    def shape_and_flatten(lst):
        if isinstance(lst, (float, int)):
            return [float(lst)], (1,)
        if isinstance(lst, list):
            shapes = [shape_and_flatten(item) for item in lst]
            flat_data = [item for sublist, _ in shapes for item in sublist]
            sub_shapes = [shape for _, shape in shapes]
            if len(set(sub_shapes)) != 1:
                raise ValueError("All sublists must have the same shape")
            return flat_data, (len(lst),) + sub_shapes[0]
        raise ValueError(f"Unsupported type: {type(lst)}")

    data, shape = shape_and_flatten(ls)
    return _tensor(data, shape, backend, requires_grad)
