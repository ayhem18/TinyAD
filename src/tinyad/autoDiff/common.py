from abc import ABC, abstractmethod
from typing import Optional, Union


NUM = Union[int, float]


class IllegalStateError(Exception):
    """Exception raised when an operation is attempted on a variable that is not in a valid state."""
    def __init__(self, message: str):
        super().__init__(message)
        self.error_message = message


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
    
    def add_parentheses(self) -> None:
        ends_with_parentheses = self.name[-1] == ")"
        starts_with_parentheses = self.name[0] == "(" 

        if ends_with_parentheses != starts_with_parentheses:
            raise IllegalStateError("The variable name must either contain opening and closing parentheses or none of them")

        if not ends_with_parentheses:
            self.name = "(" + self.name + ")"


