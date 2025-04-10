from copy import deepcopy
import random
from typing import Dict
import unittest
import numpy as np

from tinyad.autoDiff.common import Var
from tinyad.autoDiff.operators.binary_ops import Mult
from tinyad.parsing.parser import evaluate_expression, extend_expression, parse_no_operators2, parse_number, parse_variable
from tinyad.autoDiff.var import ConstantVar, ElementaryVar
from tinyad.tests.parsing.parser_base_test import ParserBaseTest


class TestNoOperatorBlockParser(ParserBaseTest):
    def test_parse_number(self):
        """Test parsing of different numbers from strings."""
        # Basic integer cases
        self.assertEqual(parse_number("123"), (123.0, 3))
        self.assertEqual(parse_number("0"), (0.0, 1))
        self.assertEqual(parse_number("5"), (5.0, 1))
        self.assertEqual(parse_number("10"), (10.0, 2))
        
        # Float cases
        self.assertEqual(parse_number("123.45"), (123.45, 6))
        self.assertEqual(parse_number("0.5"), (0.5, 3))
        self.assertEqual(parse_number("7.25"), (7.25, 4))
        self.assertEqual(parse_number("0.75"), (0.75, 4))  # No leading zero
                
        # Numbers followed by other characters
        self.assertEqual(parse_number("123x"), (123.0, 3))
        self.assertEqual(parse_number("45.67abc"), (45.67, 5))
        self.assertEqual(parse_number("8.9+10"), (8.9, 3))
        
        # Numbers with underscore (not valid in floating point literals)
        self.assertEqual(parse_number("1_000"), (1.0, 1))
        
        # Invalid inputs
        with self.assertRaises(ValueError, msg="The input should be a number"):
            parse_number("abc")
        with self.assertRaises(ValueError, msg="The input should be a number"):
            parse_number("")
        with self.assertRaises(ValueError, msg="The input should be a number"):
            parse_number(".")  # Just a decimal point   

        with self.assertRaises(ValueError):
            parse_number(".75")

        # Edge cases - very large/small numbers
        self.assertEqual(parse_number("1234567890123456"), (1234567890123456.0, 16))
        self.assertEqual(parse_number("0.0000000001"), (1e-10, 12))


    def test_parse_variable(self):
        """Test parsing of different variable formats from strings."""
        # Single letter variables
        self.assertEqual(parse_variable("x"), ("x", 1))
        self.assertEqual(parse_variable("y"), ("y", 1))
        self.assertEqual(parse_variable("Z"), ("Z", 1))
        
        # Letter followed by underscore and number
        self.assertEqual(parse_variable("x_1"), ("x_1", 3))
        self.assertEqual(parse_variable("y_23"), ("y_23", 4))
        self.assertEqual(parse_variable("Z_456"), ("Z_456", 5))
        
        # Variables followed by other characters
        self.assertEqual(parse_variable("a+b"), ("a", 1))
        self.assertEqual(parse_variable("x_2*y"), ("x_2", 3))
        self.assertEqual(parse_variable("m_10/n_5"), ("m_10", 4))
        
        # Variables that are part of a larger expression
        self.assertEqual(parse_variable("a123"), ("a", 1))  # Only 'a' is the variable
        self.assertEqual(parse_variable("b_12xy"), ("b_12", 4))
        
        # Edge cases
        with self.assertRaises(ValueError):
            parse_variable("")  # Empty string
        with self.assertRaises(ValueError):
            parse_variable("123")  # Starts with number
        with self.assertRaises(ValueError):
            parse_variable("_abc")  # Starts with underscore
        with self.assertRaises(ValueError):
            parse_variable("+x")  # Starts with symbol
            
        # Underscore without number
        self.assertEqual(parse_variable("x_"), ("x", 1))  # Should only return 'x'
        
        # Underscore followed by non-digit
        self.assertEqual(parse_variable("x_a"), ("x", 1))  # Should only return 'x'
        
        # Multiple underscores
        self.assertEqual(parse_variable("x_1_2"), ("x_1", 3))  # Should only parse up to first number
        
        # Case with special characters
        with self.assertRaises(ValueError):
            parse_variable("$var")  # Special character
            
        # Very long variable name
        long_var = "z_" + "9" * 100  # z_99...99 (100 nines)
        self.assertEqual(parse_variable(long_var), (long_var, len(long_var)))


        # Create trackers for testing


    def run_test(self, expression: str, var_name_tracker: Dict[str, Var]):
        variables, var_position_tracker = parse_no_operators2(expression, var_name_tracker)
        return variables, var_position_tracker


    def test_parse_no_operators_random(self):
        """Test parse_no_operators with many randomly generated valid expressions."""
        for _ in range(5000):  
            # Generate a random expression  
            expression, expected_vars, expected_positions = self.generate_random_expression()
            
            var_name_tracker = {}

            # Parse the expression
            variables, var_positions = self.run_test(expression, var_name_tracker)

            # make sure the position tracking is correct            
            self.assertEqual(len(var_positions), len(expected_positions))
            for i, pos in enumerate(sorted(var_positions, key=lambda x: x[0])):
                self.assertEqual(pos, expected_positions[i])


            # Verify that all variables were correctly parsed
            self.assertEqual(len(variables), len(expected_vars))
            
            # Check that all expected variables are in the name tracker 
            for var_name in expected_vars:
                self.assertIn(var_name, var_name_tracker)
            

    def test_parse_no_operators_numeric_only(self): 
        """Test parse_no_operators with expressions containing only numerical values."""
        for _ in range(5000):  
            # Generate a random number (integer or float)
            if random.random() > 0.5:
                # Integer
                num = random.randint(1, 10000)
                num_str = str(num)
            else:
                # Float with 1-3 decimal places
                num = random.uniform(0.1, 1000.0)
                decimal_places = random.randint(1, 3)
                num = round(num, decimal_places)
                num_str = str(num)
            
            name_tracker = {}
            
            # Parse the expression
            variables, var_positions = self.run_test(num_str, name_tracker)
            
            # Verify results
            self.assertEqual(len(name_tracker), 1)  # No variables
            self.assertEqual(len(var_positions), 1)   # Only the number position
            self.assertIsInstance(variables[0], ConstantVar)
            self.assertAlmostEqual(variables[0].value, float(num_str))
            
            # Verify position tracking
            self.assertTrue(len(var_positions) == 1 and (0, len(num_str)) in var_positions)
    

    def test_parse_no_operators_variables_only(self):   
        """Test parse_no_operators with expressions containing only variables (no numbers)."""
        for _ in range(5000):  
            # Generate a random expression with only variables
            expression, expected_vars, expected_positions = self.generate_random_expression(
                include_number=False, 
                num_variables=random.randint(1, 15)
            )
            
            # Parse the expression
            var_name_tracker = {}   
            variables, var_positions = self.run_test(expression, var_name_tracker)

            # make sure the position tracking is correct            
            self.assertEqual(len(var_positions), len(expected_positions))
            for i, pos in enumerate(sorted(var_positions, key=lambda x: x[0])):
                self.assertEqual(pos, expected_positions[i])

            # Verify that all variables were correctly parsed
            self.assertEqual(len(variables), len(expected_vars))
            
            # Check that all expected variables are in the name tracker
            for var_name in expected_vars:
                self.assertIn(var_name, var_name_tracker)
            

    def test_same_var_object_for_same_name(self):
        """Test that the same variable object is returned for the same name."""
        for _ in range(20000):
            # Generate a random expression with only variables
            final_var_names = []
            n1 = random.randint(2, 10)
            n2 = random.randint(2, 10)
            
            var_names = [self.generate_random_variable_name(max_underscore_num=9) for _ in range(n1)]
            var_names = list(set(var_names))
            n1 = len(var_names)

            for _ in range(n2):
                final_var_names.extend(var_names)

            # generate a random expression with the variables
            expression, expected_vars, expected_positions = self.generate_random_expression(
                include_number=False, 
                num_variables=None,
                given_var_names=final_var_names
            )

            var_name_tracker = {}
            variables, var_positions = self.run_test(expression, var_name_tracker)

            # make sure the position tracking is correct            
            self.assertEqual(len(var_positions), len(expected_positions))
            for i, pos in enumerate(sorted(var_positions, key=lambda x: x[0])):
                self.assertEqual(pos, expected_positions[i])

            # verify that all variables are the same object
            self.assertEqual(len(var_name_tracker), len(var_names))
            
            for var_name in expected_vars:
                self.assertIn(var_name, var_name_tracker)
             
            for i in range(len(var_names)):
                v = variables[i]
                for j in range(n2):
                    self.assertIs(v, variables[i + j * len(var_names)])

            
class TestParserLevel2Expressions(ParserBaseTest):
    def test_extend_expression_only_vars(self):
        """Test extend_expression with no precomputed variables."""
        
        for _ in range(20000): 
            # 1. Generate a set of random variables
            num_vars = random.randint(3, 8)
            variables = [self.generate_random_variable_name().lower() for _ in range(num_vars)]

            # 2. generate the explicit expression
            explicit_expression, exp_indices = self.generate_random_expression_with_operators(variables)
        
            # 3. generate the implicit expression
            implicit_expression = self.generate_inexplicit_expression_from_explicit(explicit_expression, num_removal=5, ops_indices=exp_indices)

            # 4. Call extend_expression
            precomputed = {}
            global_var_name_tracker = {}
            
            new_expression, variables_list, _ = extend_expression(
                implicit_expression, precomputed, global_var_name_tracker
            )

            # the returned expression must be the same as the explicit expression
            self.assertEqual(new_expression, explicit_expression) 

            # the variables list must be the same as the explicit variables
            self.assertEqual(sorted([v.name for v in variables_list]), sorted(variables))


    def test_extend_expression_precomputed_v1(self):
        for _ in range(20000):
            n_precomputed = random.randint(2, 10)
            precomputed = [self.generate_simple_precomputed_expression() for _ in range(n_precomputed)] 

            final_explicit_expression = "(" + precomputed[0].name + ")"
            final_implicit_expression = deepcopy(final_explicit_expression)

            # make sure to save the indices of the precomputed variables
            precomputed_indices = {(0, len(final_implicit_expression) - 1): precomputed[0]}

            other_var_names = []

            for i in range(1, n_precomputed):
                op = random.choice(['+', '-', '*', '/'])
                final_implicit_expression += op 
                final_explicit_expression += op 

                vars = [self.generate_random_variable_name(max_underscore_num=10) for _ in range(random.randint(2, 10))]
                other_var_names.append(vars)
                expression, _ = self.generate_random_expression_with_operators(vars)

                # create an implicit expression
                implicit_expression = self.generate_inexplicit_expression_from_explicit(expression, num_removal=random.randint(2, 5))

                final_implicit_expression += implicit_expression
                final_explicit_expression += expression

                op = random.choice(['+', '-', '*', '/'])
                final_implicit_expression += op 
                final_explicit_expression += op 

                i1 = len(final_implicit_expression)

                final_implicit_expression += ("(" + precomputed[i].name + ")")    
                final_explicit_expression += ("(" + precomputed[i].name + ")")

                i2 = len(final_implicit_expression)-1

                precomputed_indices[(i1, i2)] = precomputed[i]


            for (i1, i2), v in precomputed_indices.items():
                self.assertEqual(final_implicit_expression[i2], ")")
                self.assertEqual(final_implicit_expression[i1], "(")
                self.assertEqual(final_implicit_expression[i1 + 1:i2], v.name)


            # now call extend_expression
            new_expression, variables_list, new_precomputed = extend_expression(
                final_implicit_expression, precomputed_indices, {}
            )

            # the new expression must be the same as the explicit expression
            self.assertEqual(new_expression, final_explicit_expression)

            self.assertEqual(len(variables_list), sum([len(inner_vars) for inner_vars in other_var_names]) + n_precomputed)

            # it is important to check that the variables list is correct

            precomputed_indices = [0]
            other_vars_indices = []

            for _, v in zip(precomputed, other_var_names):
                other_vars_indices.append([precomputed_indices[-1] + i for i in range(1, len(v) + 1)])
                precomputed_indices.append(precomputed_indices[-1] + len(v) + 1)

            for order, i in enumerate(precomputed_indices):
                self.assertIs(variables_list[i], precomputed[order])

            for order, i in enumerate(other_vars_indices):
                actual = [variables_list[j].name.lower() for j in i]
                expected = [var_name.lower() for var_name in other_var_names[order]]
                expected2 = ["(" + var_name.lower() + ")" for var_name in other_var_names[order]]
                self.assertTrue(actual == expected or actual == expected2)

            # it is important to make sure that the new precomputed variables are also correct
            for order, ((i1, i2), v) in enumerate(sorted(new_precomputed.items(), key=lambda x: x[0][0])):
                self.assertEqual(new_expression[i2], ")")
                self.assertEqual(new_expression[i1], "(")
                self.assertEqual(new_expression[i1:i2+1], v.name)
                
                self.assertEqual(v, precomputed[order])


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    unittest.main()