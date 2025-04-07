import numpy as np

from abc import abstractmethod
from typing import Optional

from ..common import NUM
from ..var import Var


class UnaryOp(Var):
    def __init__(self, name: str, operand: Var):
        super().__init__(name)
        self.operand = operand
        self.children = [operand]

    @abstractmethod
    def forward(self) -> "Var":
        pass

    @abstractmethod
    def backward(self, value: Optional[NUM] = None):
        pass

    @abstractmethod
    def compute(self) -> NUM:
        pass


class Neg(UnaryOp):
    def __init__(self, operand: Var):
        super().__init__("-", operand)

    def forward(self) -> "Var":
        return Neg(self.operand)

    def backward(self, value: Optional[NUM] = None):
        if value is None:
            value = 1
        
        # For y = -x:
        # dy/dx = -1
        self.grad = value
        
        # Propagate gradient to the operand with negative sign
        self.operand.backward(-value)

    def compute(self) -> NUM:
        if self.value is not None:
            return self.value
        
        self.value = -self.operand.compute()
        return self.value


class AbsVal(UnaryOp):
    def __init__(self, operand: Var):
        super().__init__("|.|", operand)

    def forward(self) -> "Var":
        return AbsVal(self.operand)

    def backward(self, value: Optional[NUM] = None):
        if value is None:
            value = 1
        
        self.grad = value
        
        # Calculate the derivative based on the operand's value
        operand_value = self.operand.compute()

        derivative = np.sign(operand_value).item()
        
        # Propagate the gradient
        self.operand.backward(value * derivative)

    def compute(self) -> NUM:
        if self.value is not None:
            return self.value
        
        self.value = abs(self.operand.compute())
        return self.value
