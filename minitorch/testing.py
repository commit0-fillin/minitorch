from typing import Callable, Generic, Iterable, Tuple, TypeVar
import minitorch.operators as operators
A = TypeVar('A')

class MathTest(Generic[A]):

    @staticmethod
    def neg(a: A) -> A:
        """Negate the argument"""
        return operators.neg(a)

    @staticmethod
    def addConstant(a: A) -> A:
        """Add constant to the argument"""
        return operators.add(a, 5.0)

    @staticmethod
    def square(a: A) -> A:
        """Manual square"""
        return operators.mul(a, a)

    @staticmethod
    def cube(a: A) -> A:
        """Manual cube"""
        return operators.mul(operators.mul(a, a), a)

    @staticmethod
    def subConstant(a: A) -> A:
        """Subtract a constant from the argument"""
        return operators.add(a, -5.0)

    @staticmethod
    def multConstant(a: A) -> A:
        """Multiply a constant to the argument"""
        return operators.mul(a, 5.0)

    @staticmethod
    def div(a: A) -> A:
        """Divide by a constant"""
        return operators.mul(a, 0.2)  # Equivalent to dividing by 5

    @staticmethod
    def inv(a: A) -> A:
        """Invert after adding"""
        return operators.inv(operators.add(a, 1.0))

    @staticmethod
    def sig(a: A) -> A:
        """Apply sigmoid"""
        return operators.sigmoid(a)

    @staticmethod
    def log(a: A) -> A:
        """Apply log to a large value"""
        return operators.log(operators.add(a, 100.0))

    @staticmethod
    def relu(a: A) -> A:
        """Apply relu"""
        return operators.relu(a)

    @staticmethod
    def exp(a: A) -> A:
        """Apply exp to a smaller value"""
        return operators.exp(operators.mul(a, 0.1))

    @staticmethod
    def add2(a: A, b: A) -> A:
        """Add two arguments"""
        return operators.add(a, b)

    @staticmethod
    def mul2(a: A, b: A) -> A:
        """Mul two arguments"""
        return operators.mul(a, b)

    @staticmethod
    def div2(a: A, b: A) -> A:
        """Divide two arguments"""
        return operators.mul(a, operators.inv(b))

    @classmethod
    def _tests(cls) -> Tuple[Tuple[str, Callable[[A], A]], ...]:
        """
        Returns a list of all the math tests.
        """
        one_arg = [
            ("neg", cls.neg),
            ("addConstant", cls.addConstant),
            ("square", cls.square),
            ("cube", cls.cube),
            ("subConstant", cls.subConstant),
            ("multConstant", cls.multConstant),
            ("div", cls.div),
            ("inv", cls.inv),
            ("sig", cls.sig),
            ("log", cls.log),
            ("relu", cls.relu),
            ("exp", cls.exp)
        ]
        two_arg = [
            ("add2", cls.add2),
            ("mul2", cls.mul2),
            ("div2", cls.div2)
        ]
        return tuple(one_arg + two_arg)

class MathTestVariable(MathTest):
    pass
