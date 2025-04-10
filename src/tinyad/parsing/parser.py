"""
This module contains a simple parser of mathematical expressions.
"""


import re

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from copy import deepcopy

from tinyad.autoDiff.common import NUM
from tinyad.autoDiff.operators.binary_ops import Add, Div, Exp, Mult, Sub
from tinyad.autoDiff.operators.unary_ops import Neg
from tinyad.autoDiff.var import ConstantVar, ElementaryVar, Var


DEFAULT_SUPPORTED_OPERATORS = ['*', '+', '-', '/', '^']



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


# def parse_no_operators(expression: str,
#                        var_position_tracker: Dict[Tuple[int, int], Var],
#                        var_name_tracker: Dict[str, Var]
#                        ) -> Var:

#     """
#     This function parses a mathematical expression that does not contain any operators and converts it into a Var object.

#     Args:
#         expression: A string representing a mathematical expression.

#     Returns:
#         A Var object.
#     """
#     # the first step is to make sure that the expression includes only variables and numbers
#     if re.match(r'^[a-zA-Z0-9_\.]+$', expression) is None:
#         raise ValueError("The expression must include only variables and numbers")
    
#     # the second step is to parse the expression
#     # we need to find the first number in the expression (if any)
#     number_var = None
    
#     try:
#         number, i = parse_number(expression)
#         number_var = ConstantVar(str(number),number)
#         var_position_tracker[(0, i)] = number_var
#         var_name_tracker[str(number)] = number_var

#     except ValueError:
#         number = 0
#         i = 0
   
#     var_names = []
#     traverse_index = i

#     current_string = expression[traverse_index:]

#     while traverse_index < len(expression):
#         try:
#             var_name, i = parse_variable(current_string)
#         except ValueError: 
#             raise ValueError(f"a basic (no operator) expression is expected to be an optional number followed by a series of variables.")

#         if var_name in var_name_tracker:
#             var_position_tracker[(traverse_index, traverse_index + i)] = var_name_tracker[var_name]
#         else:
#             var_position_tracker[(traverse_index, traverse_index + i)] = ElementaryVar(var_name, None)
#             var_name_tracker[var_name] = var_position_tracker[(traverse_index, traverse_index + i)]
            
#         var_names.append(var_name)      
#         current_string = current_string[i:]
#         traverse_index += i

#     # the final variable is basically the product of all variables
#     if number_var is not None:
#         final_var = number_var
#     else:
#         final_var = ConstantVar(f"({0},{len(expression)})", 1)

#     for var_name in var_names:
#         final_var = Mult(final_var, var_name_tracker[var_name])

#     # var_position_tracker[(0, len(expression))] = final_var

#     return final_var

def parse_no_operators2(expression: str,
                        var_name_tracker: Dict[str, Var],
                       ) -> Tuple[List[Var], Dict[Tuple[int, int], Var]]:

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

    var_position_tracker = {}

    variables = []

    try:
        number, i = parse_number(expression)
        number_var = ConstantVar(str(number),number)
        var_position_tracker[(0, i)] = number_var
        var_name_tracker[str(number)] = number_var
        variables.append(number_var)

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
        
        variables.append(var_name_tracker[var_name])
        var_names.append(var_name)      
        current_string = current_string[i:]
        traverse_index += i


    return variables, var_position_tracker


def extend_expression(expression:str, 
                      precomputed: Dict[Tuple[int, int], Var], 
                      global_var_name_tracker: Dict[Tuple[int, int], Var]) -> Tuple[str, List[Var], Dict[Tuple[int, int], Var]]:
    """
    This function parses a mathematical expression and converts it into a Var object.
    """

    #TODO: add some code to make sure the expression does not need to be processed. (otherwise, the precomputed indices will not match the resulting string)
    # exp = deepcopy(process_expression(expression).lower())
    exp = deepcopy(expression)

    #TODO: add the code to verify whether the precomputed indices are as expected.

    # the step here is variables with parentheses here. 
    for start, end in global_var_name_tracker.items():
        exp[start:end] = "&" * (end - start) 

    # now we are guaranteed that the expression is a valid expression without parentheses 
    # we can now parse the expression 

    # determine all operators 
    ops_indices = [i for i, c in enumerate(exp) if c in DEFAULT_SUPPORTED_OPERATORS]

    if len(ops_indices) == 0:
        # this means that the expression does not contain any operators
        # extract the variables, join them with the * operator and return the result
        variables, _ = parse_no_operators2(exp, global_var_name_tracker)
        return "*".join([v.name for v in variables]), variables, {}


    # the expression will be reconstructed by adding the implicit multiplication operators 
    # between the variables 
        
    new_expression = exp[:ops_indices[0]]
    new_precomputed = {}
    
    if len(new_expression) > 0:
        variables, _ = parse_no_operators2(new_expression, global_var_name_tracker)
        new_expression = "*".join([v.name for v in variables])
    else:
        variables = []


    for i in range(len(ops_indices)):
        current_idx = ops_indices[i]
        next_idx = ops_indices[i + 1] if i + 1 < len(ops_indices) else len(exp) 

        exp_no_ops = exp[current_idx + 1:next_idx]

        if (current_idx + 1, next_idx) in precomputed:
            new_precomputed[len(new_expression), len(new_expression) + len(exp_no_ops)] = precomputed[(current_idx + 1, next_idx)]
            new_expression += "&" * len(exp_no_ops)
            variables.append(precomputed[(current_idx + 1, next_idx)])
        else:
            inner_vars, var_position_tracker = parse_no_operators2(exp_no_ops, global_var_name_tracker) 
            inner_expression = "*".join([v.name for v in inner_vars])
            
            new_expression += (exp[current_idx] + inner_expression)
            variables.extend(inner_vars)

    return new_expression, variables, new_precomputed


def evaluate_expression_inner(extended_expression:str, variables: List[Var]) -> Var:
    """
    This function parses a mathematical expression and converts it into a Var object.
    """
    operator_indices_in_expression = [i for i, c in enumerate(extended_expression) if c in DEFAULT_SUPPORTED_OPERATORS]

    # the op2indices is a dictionary that maps the operator to the indices of that operator in the `extended_expression` string 
    op2indices = {"+-": [], '*/': [], '^': []}
    
    # the op2order is a dictionary that maps the operator to the order of that operator in the `operator_indices_in_expression` list 
    op2order = {"+-": [], '*/': [], '^': []}
   
    for i, c in enumerate(extended_expression):
        for op_string in op2order:
            if c in op_string:
                op2indices[op_string].append(i)

    for order, idx in enumerate(operator_indices_in_expression):
        operator = extended_expression[idx]
        for ops_string in op2order:
            if operator in ops_string:
                op2order[ops_string].append(order)

    if operator_indices_in_expression[0] == 0:
        if extended_expression[0] != '-':
            raise ValueError("if the expression starts with an operator, it must be a '-'")
        else:
            variables[0] = Neg(variables[0]) 
            operator_indices_in_expression = operator_indices_in_expression[1:]

    if len(operator_indices_in_expression) != len(variables) - 1:
        raise ValueError("something happened")

    # at this point we are ready to create the final variable:
    # the idea is a bit tricky

    exp_propagate = {}

    # 1. compute the exponential operator first
    for idx in op2order["^"]:
        res_var = Exp(variables[idx], variables[idx + 1]) 
        variables[idx] = res_var
        variables[idx + 1] = res_var
        
        # basically this is a dictionary that makes sure variables that coupled together are updated together
        exp_propagate[idx] = idx + 1
        exp_propagate[idx + 1] = idx



    # 2. find blocks of multiplication and division operators   

    mult_div_blocks = []
    last_block = []

    
    for i in range(len(operator_indices_in_expression)):
        if extended_expression[operator_indices_in_expression[i]] in ['*', '/']:
            last_block.append(i)
        else:
            if len(last_block) > 0:
                mult_div_blocks.append(last_block)
                last_block = []

    if len(last_block) > 0:
        mult_div_blocks.append(last_block)

    # 2. compute the multiplication and division operators next

    for block in mult_div_blocks: 
        res_var = variables[block[0]]

        for op_index in block:
            if extended_expression[operator_indices_in_expression[op_index]] == '*':
                res_var = Mult(res_var, variables[op_index + 1])
            else:
                res_var = Div(res_var, variables[op_index + 1])
        
        # make sure to set the new variable 
        for op_index in block:
            variables[op_index] = res_var
            if op_index in exp_propagate:
                variables[exp_propagate[op_index]] = res_var 

        variables[block[-1] + 1] = res_var        
        if block[-1] + 1 in exp_propagate:
            variables[exp_propagate[block[-1] + 1]] = res_var

    # consider the case where there is no addition or subtraction operators 
    if len(op2order["+-"]) == 0:
        return res_var

    # 3. compute the addition and subtraction operators next
    
    first_index = min(op2order["+-"])

    if extended_expression[operator_indices_in_expression[first_index]] != "+":
        raise ValueError("something happened !!!, at this point + must appear before -")

    final_var = variables[first_index]
    
    for i in op2order["+-"]:
        if extended_expression[operator_indices_in_expression[i]] == "+":
            final_var = Add(final_var, variables[i + 1])
        else:
            final_var = Sub(final_var, variables[i + 1])

    return final_var

    
def evaluate_expression(expression: str, precomputed: Dict[Tuple[int, int], Var], global_var_name_tracker: Dict[Tuple[int, int], Var]) -> Var:
    extended_expression, variables, new_precomputed = extend_expression(expression, precomputed, global_var_name_tracker)
    return evaluate_expression_inner(extended_expression, variables)




# def parse_expression_no_parentheses(expression: str, var_name_tracker: Dict[str, Var], var_position_tracker: Dict[Tuple[int, int], Var]) -> Var:
#     """
#     This function parses a mathematical expression and converts it into a Var object.
#     """
#     exp = copy(expression.lower())

#     # we assume here that the variable position tracker is adjusted to the expression length
    
#     # the step here is variables with parentheses here. 
#     for start, end in var_position_tracker.items():
#         exp[start:end] = "&" * (end - start) 

#     # now we are guaranteed that the expression is a valid expression without parentheses 
#     # we can now parse the expression 

#     # determine all operators 
#     ops_indices = [i for i, c in enumerate(exp) if c in DEFAULT_SUPPORTED_OPERATORS]

#     # xyz ^ 4


#     vars = []

#     for i in range(len(ops_indices)):
#         current_idx = ops_indices[i]
#         next_idx = ops_indices[i + 1] if i + 1 < len(ops_indices) else len(exp) 

#         exp_no_ops = exp[current_idx + 1:next_idx] 

#         if (current_idx + 1, next_idx) in var_position_tracker:
#             vars.append(var_position_tracker[(current_idx + 1, next_idx)])
#         else:
#             vars.append(parse_no_operators(exp_no_ops, var_position_tracker, var_name_tracker)) 

#     if ops_indices[0] == 0:
#         if exp[0] != '-':
#             raise ValueError("if the expression starts with an operator, it must be a '-'")
#         else:
#             vars[0] = Neg(vars[0]) 
#             ops_indices = ops_indices[1:]

#     else:
#         # this means that code above did not include the first variable
#         first_var_exp = exp[:ops_indices[0]]
#         if (0, len(first_var_exp)) in var_position_tracker:
#             vars.insert(0, var_position_tracker[(0, len(first_var_exp))])
#         else:
#             vars.insert(0, parse_no_operators(first_var_exp, var_position_tracker, var_name_tracker))

    
#     # at this point the number of operators must be exactly one less than the number of variables 
#     if len(ops_indices) != len(vars) - 1:
#         raise ValueError("something happened")

    