from abc import abstractmethod
from typing import Optional

from ..common import NUM
from ..var import Var


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
        super().__init__("+", left, right)


    def forward(self) -> "Var":
        return Add(self.left.forward(), self.right.forward())
    
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
        super().__init__("-", left, right)


    def forward(self) -> "Var":
        return Sub(self.left.forward(), self.right.forward())
    

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
        super().__init__("*", left, right)


    def forward(self) -> "Var":
        return Mult(self.left.forward(), self.right.forward())
    
    
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
    def __init__(self, left: Var, right: Var, numerical_issue_tolerance: float = 1e-8):
        super().__init__("/", left, right)
        self.numerical_issue_tolerance = numerical_issue_tolerance

        right_val = self.right.compute()

        if right_val == 0:
            raise ValueError("Division by zero")

        if abs(right_val) < numerical_issue_tolerance:
            raise ValueError("Division by a number too close to zero")


    def forward(self) -> "Var":
        return Div(self.left.forward(), self.right.forward())
    

    def backward(self, value: Optional[NUM] = None):
        if value is None:
            value = 1
        
        # For z = var1 / var2:
        # dz/dvar1 = 1/var2, dz/dvar2 = -var1/var2²
        self.grad = value
        
        right_val = self.right.compute()
        
        # propagate the gradient to the children
        self.left.backward(value / right_val)
        self.right.backward(-value * self.left.compute() / (right_val ** 2))


    def compute(self) -> NUM:
        if self.value is not None:
            return self.value
        
        self.value = self.left.compute() / self.right.compute()
        return self.value



