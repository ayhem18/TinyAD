from typing import Optional

from .common import NUM, Var, IllegalStateError


class ElementaryVar(Var):
    def __init__(self, name: str, value: NUM):
        super().__init__(name)
        self.value = value

    def forward(self) -> "Var":
        return self
    
    def backward(self, value: Optional[NUM] = None):
        if value is None:
            value = 1

        if self.grad is None:
            self.grad = value
        else:
            self.grad += value

    def compute(self) -> NUM:
        if self.value is None:
            raise IllegalStateError("Value is not set for this variable")

        return self.value
    

class ConstantVar(Var):
    def __init__(self, name: str, value: NUM):
        super().__init__(name)
        self.value = value

    def forward(self) -> "Var":
        return self
    
    def backward(self, value: Optional[NUM] = None):
        self.grad = 0

    def compute(self) -> NUM:
        return self.value   
    



    