from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from typing_extensions import Protocol

def central_difference(f: Any, *vals: Any, arg: int=0, epsilon: float=1e-06) -> Any:
    """
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \\ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \\ldots, x_{n-1})$
    """
    vals_list = list(vals)
    vals_plus_epsilon = vals_list.copy()
    vals_plus_epsilon[arg] += epsilon
    vals_minus_epsilon = vals_list.copy()
    vals_minus_epsilon[arg] -= epsilon
    
    return (f(*vals_plus_epsilon) - f(*vals_minus_epsilon)) / (2 * epsilon)
variable_count = 1

class Variable(Protocol):
    pass

def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    result = []

    def dfs(var):
        if var not in visited and not var.is_constant():
            visited.add(var)
            if hasattr(var, 'history') and var.history is not None:
                for child in var.history.inputs:
                    dfs(child)
            result.append(var)

    dfs(variable)
    return reversed(result)

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    queue = [(variable, deriv)]
    visited = set()

    while queue:
        var, der = queue.pop(0)
        if var in visited:
            continue
        visited.add(var)

        if var.is_leaf():
            var.accumulate_derivative(der)
        elif var.history:
            for inp, grad in zip(var.history.inputs, var.history.backprop_step(der)):
                queue.append((inp, grad))

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """
    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if not self.no_grad:
            self.saved_values = values
