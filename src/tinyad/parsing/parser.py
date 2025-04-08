"""
This module contains a simple parser of mathematical expressions.
"""


import re
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from tinyad.autoDiff.var import Var


def process_expression(expression: str) -> str:
    """
    Process a mathematical expression to remove parentheses and other non-essential characters.
    """
    # remove all spaces 
    expression = re.sub(r'\s+', '', expression)
    return expression   



def parse_no_parentheses(expression: str) -> Var:
    """
    Parse a mathematical expression without parentheses.
    """
    pass        


def build_parentheses_tree(expression: str) -> Dict[int, List[Tuple[Optional[int], int, int]]]:
    """
    Given a mathematical expression, this function finds all the parentheses and returns a datastructure that allows
    computing all inner expressions.

    Args:
        expression: A string representing a mathematical expression.

    Returns:
        A dictionary where each key represents a level, the value is a list of tuples. Each tuple: [i1, i2, i3] where i1 is the index of the parent in 
        the previous level, i2 the index of opening parenthesis, and i3 the index of the closing parenthesis. 
    """

    result = {}
    # create a stack to keep track of the levels and indices

    if expression[0] != "(" or expression[-1] != ")":
        raise ValueError("The expression must start and end with a parenthesis")

    stack = deque()
    
    for i, char in enumerate(expression):
        if char == "(":
            level = len(stack)

            if level not in result:
                result[level] = []

            parent = None if level - 1 not in result else len(result[level - 1]) - 1
            result[level].append((parent, i, None))

            stack.append(i)


        elif char == ")":
            
            if len(stack) == 0:
                raise ValueError("The expression has unbalanced parentheses")
            else:
                last_open_idx = stack.pop()            
                level = len(stack)
                result[level][-1] = (result[level][-1][0], last_open_idx, i)


    return result
