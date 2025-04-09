"""
This module contains a simple parser of mathematical expressions.
"""


import re

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from tinyad.autoDiff.common import NUM
from tinyad.autoDiff.operators.binary_ops import Mult
from tinyad.autoDiff.var import ConstantVar, ElementaryVar, Var


DEFAULT_SUPPORTED_OPERATORS = ['*', '+', '-', '/']


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



def parse_number(string: str) -> Tuple[NUM, int]:
    """
    This functions goes through a string and tries to extract any number at the start of the string.
    It returns the number and the index of the first character after the number.
    """
    number_str = ""
    
    i = 0
    while i < len(string):
        try:
            number_str += string[i]
            float(number_str)
            i += 1
        except ValueError:
            # this means that the last character should be removed
            number_str = number_str[:-1]
            break

    return float(number_str), i


def parse_variable(string: str) -> Tuple[str, int]:
    """
    This function goes through a string and tries to extract any variable at the start of the string.
    It returns the variable and the index of the first character after the variable.
    """
    # a variable that can two formats: either a single letter or a letter followed by an underscore and a number 
    # the number must be an integer

    if len(string) == 0:
        raise ValueError("The variable must be at least one character long")

    if not string[0].isalpha():
        raise ValueError("The variable must start with a letter")

    var = string[0] 
    
    if len(string) == 1 or string[1] != "_":
        return var, 1

    # at this point,we know that variable is of the form x_     
    i = 2
    while i < len(string) and string[i].isdigit():
        i += 1

    if i == 2:
        # this means that there are no digits after the underscore
        i = 1

    return string[:i], i


def parse_no_operators(expression: str,
                       var_position_tracker: Dict[Tuple[int, int], Var],
                       var_name_tracker: Dict[str, Var]
                       ) -> Var:

    """
    This function parses a mathematical expression that does not contain any operators and converts it into a Var object.

    Args:
        expression: A string representing a mathematical expression.

    Returns:
        A Var object.
    """
    # the first step is to make sure that the expression includes only variables and numbers
    if re.match(r'^[a-zA-Z0-9_\.]+$', expression) is None:
        raise ValueError("The expression must include only variables and numbers")
    
    # the second step is to parse the expression
    # we need to find the first number in the expression (if any)
    number_var = None
    
    try:
        number, i = parse_number(expression)
        number_var = ConstantVar(str(number),number)
        var_position_tracker[(0, i)] = number_var
        var_name_tracker[str(number)] = number_var

    except ValueError:
        number = 0
        i = 0
   
    var_names = []
    traverse_index = i

    current_string = expression[traverse_index:]

    while traverse_index < len(expression):
        try:
            var_name, i = parse_variable(current_string)
        except ValueError: 
            raise ValueError(f"a basic (no operator) expression is expected to be an optional number followed by a series of variables.")

        if var_name in var_name_tracker:
            var_position_tracker[(traverse_index, traverse_index + i)] = var_name_tracker[var_name]
        else:
            var_position_tracker[(traverse_index, traverse_index + i)] = ElementaryVar(var_name, None)
            var_name_tracker[var_name] = var_position_tracker[(traverse_index, traverse_index + i)]
            
        var_names.append(var_name)      
        current_string = current_string[i:]
        traverse_index += i

    # the final variable is basically the product of all variables
    if number_var is not None:
        final_var = number_var
    else:
        final_var = ConstantVar(f"({0},{len(expression)})", 1)

    for var_name in var_names:
        final_var = Mult(final_var, var_name_tracker[var_name])

    # var_position_tracker[(0, len(expression))] = final_var

    return final_var
