from abc import ABC, abstractmethod
from typing import Optional, Union

NUM = Union[int, float]


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
    

class IllegalStateError(Exception):
    """Exception raised when an operation is attempted on a variable that is not in a valid state."""
    def __init__(self, message: str):
        super().__init__(message)
        self.error_message = message
