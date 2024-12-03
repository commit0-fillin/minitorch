from __future__ import annotations
from typing import TYPE_CHECKING
import minitorch
from . import operators
from .autodiff import Context
if TYPE_CHECKING:
    from typing import Tuple
    from .scalar import Scalar, ScalarLike

def wrap_tuple(x):
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)

def unwrap_tuple(x):
    """Turn a singleton tuple into a value"""
    if len(x) == 1:
        return x[0]
    return x

class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

class Mul(ScalarFunction):
    """Multiplication function"""

class Inv(ScalarFunction):
    """Inverse function"""

class Neg(ScalarFunction):
    """Negation function"""

class Sigmoid(ScalarFunction):
    """Sigmoid function"""

class ReLU(ScalarFunction):
    """ReLU function"""

class Exp(ScalarFunction):
    """Exp function"""

class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

class EQ(ScalarFunction):
    """Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"""
