from typing import Optional
from abc import ABC, abstractmethod

from .common import NUM

class Var(ABC):
    def __init__(self, name: str):
        self.name = name
        self.value = None
        self.grad = None
        self.children = []
    
    def __call__(self) -> "Var":
        return self.forward()

    @abstractmethod
    def forward(self) -> "Var":
        pass

    @abstractmethod
    def backward(self, value: Optional[NUM] = None):
        pass

    @abstractmethod
    def compute(self) -> NUM:
        pass 


class ElementaryVar(Var):
    def __init__(self, name: str, value: NUM):
        super().__init__(name)
        self.value = value

    def forward(self) -> "Var":
        return self
    
    def backward(self, value: Optional[NUM] = None):
        if self.grad is None:
            self.grad = value
        else:
            self.grad += value

    def compute(self) -> NUM:
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
    



    