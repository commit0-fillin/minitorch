from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union
import numpy as np
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import EQ, LT, Add, Exp, Inv, Log, Mul, Neg, ReLU, ScalarFunction, Sigmoid
ScalarLike = Union[float, int, 'Scalar']

@dataclass
class ScalarHistory:
    """
    `ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """
    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()
_var_count = 0

class Scalar:
    """
    A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """
    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(self, v: float, back: ScalarHistory=ScalarHistory(), name: Optional[str]=None):
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        return 'Scalar(%f)' % self.data

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        raise NotImplementedError('Need to implement for Task 1.2')

    def __bool__(self) -> bool:
        return bool(self.data)

    def __lt__(self, b: ScalarLike) -> Scalar:
        raise NotImplementedError('Need to implement for Task 1.2')

    def __gt__(self, b: ScalarLike) -> Scalar:
        raise NotImplementedError('Need to implement for Task 1.2')

    def __eq__(self, b: ScalarLike) -> Scalar:
        raise NotImplementedError('Need to implement for Task 1.2')

    def __sub__(self, b: ScalarLike) -> Scalar:
        raise NotImplementedError('Need to implement for Task 1.2')

    def __neg__(self) -> Scalar:
        raise NotImplementedError('Need to implement for Task 1.2')

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x: value to be accumulated
        """
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history.last_fn is None

    def backward(self, d_output: Optional[float]=None) -> None:
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(Variable(self), d_output)

def derivative_check(f: Any, *scalars: Scalar) -> None:
    """
    Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters:
        f : function from n-scalars to 1-scalar.
        *scalars  : n input scalar values.
    """
    out = f(*scalars)
    out.backward()
    
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(f"Derivative at position {i}: {x.derivative}")
        print(f"Central difference at position {i}: {check}")
        assert abs(x.derivative - check) < 1e-3
