import random
from typing import List, Optional, Tuple
import unittest

from tinyad.autoDiff.common import Var
from tinyad.autoDiff.operators.binary_ops import Add, Mult, Sub, Exp, Div
from tinyad.autoDiff.var import ConstantVar, ElementaryVar


class ParserBaseTest(unittest.TestCase):

    def generate_random_variable_name(self, max_underscore_num: int = 999) -> str:
        """Generate a random variable name."""
        # Generate a random letter (a-z, A-Z)
        
        letter = random.choice("abcdefghijklmnopqrstuvwxyz")
        var_name = letter
        
        # Optionally add an underscore and a number
        if random.random() > 0.6:
            var_name += "_"
            var_name += str(random.randint(1, max_underscore_num))

        return var_name


    def generate_random_expression(self, 
                                    include_number=True, 
                                    num_variables=None,
                                    given_var_names: Optional[List[str]] = None):
        """Generate a random valid expression without operators.
        
        Returns:
            tuple: (expression, variable_names, variable_positions)
            - expression: the generated expression string
            - variable_names: list of variable names in order of appearance
            - variable_positions: list of (start, end) positions for each variable
        """

        expression = ""

        if given_var_names is not None:
            variable_names = given_var_names.copy()
            num_variables = len(variable_names)
        else:
            variable_names = []
        
        variable_positions = []
        
        # Step 1: Optionally add a number at the beginning
        if include_number and random.random() > 0.2:
            # Generate a random number (integer or float)
            if random.random() > 0.5:
                # Integer
                num = float(random.randint(1, 1000))
                num_str = str(num)
            else:
                # Float with 1-3 decimal places
                num = random.uniform(0.1, 1000.0)
                decimal_places = random.randint(1, 3)
                num = round(num, decimal_places)
                num_str = str(num)

            expression += num_str
            
            if given_var_names is None:
                variable_names.append(num_str)
                variable_positions.append((0, len(num_str)))
            else:
                variable_names.insert(0, num_str)
                variable_positions.insert(0, (0, len(num_str)))

        # Step 2: Generate a random number of variables if not specified
        if num_variables is None:
            num_variables = random.randint(1, 10)
        
        var_names_set = set()

        # Step 3: Generate random variables
        for i in range(num_variables):
            var_start = len(expression)
            
            if given_var_names is None:
                vn = self.generate_random_variable_name()
                
                while vn in var_names_set:
                    vn = self.generate_random_variable_name()
            else:
                vn = given_var_names[i]

            expression += vn
            variable_names.append(vn)
            var_names_set.add(vn)
            variable_positions.append((var_start, var_start + len(vn)))
        
        return expression, variable_names, variable_positions


    def generate_random_expression_with_operators(self, var_names: List[str]) -> Tuple[str, List[int]]:
        """
        Generate a random valid expression with operators.

        Args:
            var_names: list of variable names in order of appearance

        Returns:
            tuple: (expression, operators)
            - expression: the generated expression string
            - operators: list of operators in order of appearance
        """
        # 2. Create an expression with explicit operators
        operators = ['+', '-', '*', '/', '^']
        
        exp_ops = []
        
        # Start with first variable
        expression = var_names[0]
        
        # Add operator and variable pairs
        for i in range(1, len(var_names)):
            exp_ops.append(len(expression))
            # Select a random operator
            op = random.choice(operators)
            expression += op + var_names[i]

        return expression, exp_ops


    def generate_inexplicit_expression_from_explicit(self, 
                                                     explicit_expression: str, 
                                                     num_removal:int,
                                                     ops_indices: Optional[List[int]] = None) -> str:
        """
        Generate an expression with implicit operators from an expression with explicit operators.
        """
        if ops_indices is None:
            ops_indices = [i for i, char in enumerate(explicit_expression) if char in ['+', '-', '*', '/', '^']]
 
        mult_indices = [i for i in ops_indices if explicit_expression[i] == '*']
        
        num_removal = min(num_removal, len(mult_indices))        
        removal_indices = random.sample(mult_indices, num_removal)

        explicit_expression = list(explicit_expression)
        for i in removal_indices:
            explicit_expression[i] = ' '

        return ''.join(explicit_expression).replace(' ', '')
        

    def generate_simple_precomputed_expression(self) -> Var:
        """
        Generate a precomputed expression with implicit operators.
        """
        
        vn1 = self.generate_random_variable_name(max_underscore_num=10).lower()
        vn2 = self.generate_random_variable_name(max_underscore_num=10).lower()
        
        operators = ['+', '-', '*', '/', '^']
        
        op = random.choice(operators)

        if op == '*':
            return Mult(ElementaryVar(vn1, None), ElementaryVar(vn2, None))
        elif op == '/':
            return Div(ElementaryVar(vn1, None), ElementaryVar(vn2, None))
        elif op == '^':
            x = round(random.uniform(1, 100),2)
            return Exp(ElementaryVar(vn1, None), ConstantVar(str(x), x))
        elif op == '+':
            return Add(ElementaryVar(vn1, None), ElementaryVar(vn2, None))
        else:
            return Sub(ElementaryVar(vn1, None), ElementaryVar(vn2, None))

        