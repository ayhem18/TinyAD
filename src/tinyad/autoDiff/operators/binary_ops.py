from abc import abstractmethod
from typing import Optional, Tuple, Union
import warnings

import numpy as np

from ..common import NUM, IllegalStateError, Var
from ..var import ConstantVar


class BinaryOp(Var):
    def __init__(self, name: str, left: Var, right: Var):
        super().__init__(name)
        self.left = left
        self.right = right
        self.children = [left, right]

    @abstractmethod
    def forward(self) -> "Var":
        pass

    @abstractmethod
    def backward(self, value: Optional[NUM] = None):
        pass

    @abstractmethod
    def compute(self) -> NUM:
        pass



class Add(BinaryOp):
    def __init__(self, left: Var, right: Var):
        super().__init__(f"{left.name}+{right.name}", left, right)


    def forward(self) -> "Var":
        return Add(self.left, self.right)
    
    def backward(self, value: Optional[NUM] = None):
        if value is None:
            value = 1
        # self.grad saves the derivative of the output with the respect to the current variable
        
        # if L is another function that uses the current variable (let's call it z) and z = var1 + var2
        # then dL/dz = value and dL/dvar1 = value * dvar1/dz and dL/dvar2 = value * dvar2/dz

        self.grad = value

        # propagate the gradient to the children
        self.left.backward(value)
        self.right.backward(value)


    def compute(self) -> NUM:
        if self.value is not None:
            return self.value
        
        # do the computation once and save it 
        self.value = self.left.compute() + self.right.compute()
        return self.value
    

class Sub(BinaryOp):
    def __init__(self, left: Var, right: Var):
        super().__init__(f"{left.name}-{right.name}", left, right)


    def forward(self) -> "Var":
        return Sub(self.left, self.right)
    

    def backward(self, value: Optional[NUM] = None):
        if value is None:
            value = 1
        
        # For z = var1 - var2:
        # dz/dvar1 = 1, dz/dvar2 = -1
        self.grad = value
        
        # propagate the gradient to the children
        self.left.backward(value)
        self.right.backward(-value)  # Negative gradient for the right operand


    def compute(self) -> NUM:
        if self.value is not None:
            return self.value
        
        self.value = self.left.compute() - self.right.compute()
        return self.value


class Mult(BinaryOp):
    def __init__(self, left: Var, right: Var):
        super().__init__(f"{left.name}*{right.name}", left, right)


    def forward(self) -> "Var":
        return Mult(self.left, self.right)
    
    
    def backward(self, value: Optional[NUM] = None):
        if value is None:
            value = 1
        
        # For z = var1 * var2:
        # dz/dvar1 = var2, dz/dvar2 = var1
        self.grad = value
        
        # propagate the gradient to the children
        self.left.backward(value * self.right.compute())
        self.right.backward(value * self.left.compute())

    def compute(self) -> NUM:
        if self.value is not None:
            return self.value
        
        self.value = self.left.compute() * self.right.compute()
        return self.value


class Div(BinaryOp):
    def __value_checks(self, left: Var, right: Var) -> Tuple[NUM, NUM]:
        if isinstance(right, Var):
            right = right.compute()
        
        if right == 0:
            raise ValueError("Division by zero")

        if abs(right) < self.numerical_issue_tolerance:
            raise ValueError("Division by a number too close to zero")

        if isinstance(left, Var):
            left = left.compute()

        return left, right

    def __init__(self, left: Var, right: Var, numerical_issue_tolerance: float = 1e-10):
        super().__init__(f"{left.name}/{right.name}", left, right)
        self.numerical_issue_tolerance = numerical_issue_tolerance

        try:
            right_val = self.right.compute()

            if right_val == 0:
                raise ValueError("Division by zero")

            if abs(right_val) < numerical_issue_tolerance:
                raise ValueError("Division by a number too close to zero")

        except IllegalStateError:
            # this means that the right variable is not set yet
            pass



    def forward(self) -> "Var":
        return Div(self.left, self.right)
    

    def backward(self, value: Optional[NUM] = None):
        if value is None:
            value = 1
        
        # For z = var1 / var2:
        # dz/dvar1 = 1/var2, dz/dvar2 = -var1/var2²
        self.grad = value
        
        left_val, right_val = self.__value_checks(self.left, self.right)
        
        # propagate the gradient to the children
        self.left.backward(value / right_val)
        self.right.backward(-value * left_val / (right_val ** 2))


    def compute(self) -> NUM:
        if self.value is not None:
            return self.value
        
        left_val, right_val = self.__value_checks(self.left, self.right)
        self.value = left_val / right_val
        return self.value


class Exp(BinaryOp):

    def __value_checks(self, base: Union[Var, NUM], exponent: Union[Var, NUM]) -> Tuple[NUM, NUM]:
        # calling these methods triggers the computation of the value:
        # which means, it assumes the value is already set !!!
        if isinstance(base, Var):
            base = base.compute()

        if isinstance(exponent, Var):
            exponent = exponent.compute()
        
        # if exponent == 0:
        #     raise ValueError("The the exponent must be set to a value different from 0")

        if base == 0 and exponent < 0:
            raise ValueError("The base must be set to a value different from 0 when the exponent is less or equal to 0")

        return base, exponent



    def __init__(self, base: Var, exponent: Var):
        super().__init__(f"{base.name}^{exponent.name}", base, exponent)


        try:
            if abs(base.compute()) < 1e-8:
                warnings.warn("The base is too close to zero. This might be lead to numerical issues")
        except IllegalStateError:
            # this means that the base is not set yet.
            pass 

        try:
            self.exponent_value = exponent.compute()

            if self.exponent_value == 0:
                raise ValueError("The the exponent must be set to a value different from 0")

        except IllegalStateError:
            # this means that the exponent is not set yet.
            pass


    def forward(self) -> "Var":
        return Exp(self.left, self.right)
    
    def backward(self, value: Optional[NUM] = None):
        if value is None:
            value = 1
        
        # For y = x^n:
        # dy/dx = n * x^(n-1)
        self.grad = value
        
        base_value, exponent = self.__value_checks(self.left, self.right)
        
        # Calculate gradient for the base
        # Guard against potential numerical issues
        if base_value == 0:
            gradient_base = 0  # Handle 0^n cases carefully
        else:
            gradient_base = exponent * (base_value ** (exponent - 1))
        
        # Calculate gradient for the exponent

        if base_value == 0:
            gradient_exponent = 0
        else:
            gradient_exponent = np.log(base_value).item() * (base_value ** (exponent))

        # Propagate gradient to the base (left operand)
        self.left.backward(value * gradient_base)
        
        # No gradient propagation to the exponent (constant)
        self.right.backward(value * gradient_exponent)

    def compute(self) -> NUM:
        if self.value is not None:
            return self.value

        # make sure the values satisfy the constraints
        base, exponent = self.__value_checks(self.left, self.right)

        if base == 0:
            return 0

        self.value = base ** exponent
        return self.value



