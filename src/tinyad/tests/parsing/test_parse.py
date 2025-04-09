import random
import unittest

import numpy as np

from tinyad.parsing.parser import parse_number, parse_variable, parse_no_operators
from tinyad.autoDiff.var import ConstantVar, ElementaryVar
from tinyad.autoDiff.operators.binary_ops import Mult


class TestParser(unittest.TestCase):
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


    def run_test(self, expression, expected_value=None):
        var_position_tracker = {}
        var_name_tracker = {}
        result = parse_no_operators(expression, var_position_tracker, var_name_tracker)
        
        if expected_value is not None:
            self.assertAlmostEqual(result.compute(), expected_value)
        
        # Return for additional checks
        return result, var_position_tracker, var_name_tracker


    # def test_parse_no_operators(self):
    #     """Test parsing of expressions without operators."""
        
        
    #     # Test 1: Simple expression with number and variables
    #     result, pos_tracker, name_tracker = self.run_test("2xy")
    #     # Verify the result is correctly formed as 2*x*y
    #     self.assertEqual(len(name_tracker), 2)  # Should track x and y
    #     self.assertEqual(len(pos_tracker), 4)   # Start pos, 2, x, y
    #     self.assertIn("x", name_tracker)
    #     self.assertIn("y", name_tracker)
        
    #     # Test 2: Expression with only variables
    #     result, pos_tracker, name_tracker = self.run_test("abc")
    #     self.assertEqual(len(name_tracker), 3)  # Should track a, b, c
    #     self.assertIn("a", name_tracker)
    #     self.assertIn("b", name_tracker)
    #     self.assertIn("c", name_tracker)
        
    #     # Test 3: Expression with complex variable names
    #     result, pos_tracker, name_tracker = self.run_test("3x_1y_2z")
    #     self.assertEqual(len(name_tracker), 3)  # Should track x_1, y_2, z
    #     self.assertIn("x_1", name_tracker)
    #     self.assertIn("y_2", name_tracker)
    #     self.assertIn("z", name_tracker)
        
    #     # Test 4: Expression with just a number
    #     result, pos_tracker, name_tracker = self.run_test("42")
    #     self.assertEqual(len(name_tracker), 0)  # No variables
    #     self.assertEqual(result.value, 42.0)
        
    #     # Test 5: Expression with decimal number
    #     result, pos_tracker, name_tracker = self.run_test("3.14abc")
    #     self.assertEqual(len(name_tracker), 3)  # a, b, c
    #     self.assertIn((0, 3), pos_tracker)
    #     self.assertIn("a", name_tracker)
    #     self.assertIn("b", name_tracker)
    #     self.assertIn("c", name_tracker)
        
    #     # Test 6: Check structure of the expression tree
    #     # For expression like "2xy", we should get a structure:
    #     # Mult(Mult(ConstantVar(2), ElementaryVar("x")), ElementaryVar("y"))
    #     result, _, _ = self.run_test("2xy")
    #     self.assertIsInstance(result, Mult)
    #     self.assertIsInstance(result.left, Mult)
    #     self.assertIsInstance(result.left.left, ConstantVar)
    #     self.assertEqual(result.left.left.value, 2.0)
    #     self.assertIsInstance(result.left.right, ElementaryVar)
    #     self.assertEqual(result.left.right.name, "x")
    #     self.assertIsInstance(result.right, ElementaryVar)
    #     self.assertEqual(result.right.name, "y")
        
    #     # Test 7: Variable tracking with complex expressions
    #     result, pos_tracker, _ = self.run_test("5.2a_1bc_2")
    #     # Check that the position tracker has the right entries
    #     self.assertIn((0, 3), pos_tracker)  # 5.2
    #     # Get position based on the expression structure
    #     a1_pos = 3
    #     a1_end = a1_pos + 3  # "a_1" is 3 chars
    #     b_pos = a1_end
    #     b_end = b_pos + 1   # "b" is 1 char
    #     c2_pos = b_end
    #     c2_end = c2_pos + 3  # "c_2" is 3 chars
        
    #     self.assertIn((a1_pos, a1_end), pos_tracker)  # a_1
    #     self.assertIn((b_pos, b_end), pos_tracker)    # b
    #     self.assertIn((c2_pos, c2_end), pos_tracker)  # c_2
        
    #     # Test 8: Error cases
    #     with self.assertRaises(ValueError):
    #         self.run_test("xy+z")  # Contains invalid character '+'
        
    #     with self.assertRaises(ValueError):
    #         self.run_test("2x&y")  # Contains invalid character '&'
        
    #     # Test 9: Edge case - empty string
    #     with self.assertRaises(ValueError):
    #         self.run_test("")
        
    #     # Test 10: Very long expression
    #     long_var_expr = "9" + "a" * 50 + "b_" + "9" * 10
    #     result, pos_tracker, name_tracker = self.run_test(long_var_expr)
    #     # Should have 9, a (50 times), and b_9999999999
    #     self.assertEqual(len(name_tracker), 51)  # 50 'a's + 1 'b_9999999999'
        
    #     # Test 11: Check numerical computation
    #     # For "2xy" with x=3 and y=4, we should get 2*3*4 = 24
    #     result, _, name_tracker = self.run_test("2xy")
    #     # Assign values to the variables
    #     name_tracker["x"].value = 3.0
    #     name_tracker["y"].value = 4.0
    #     # Compute the expression
    #     self.assertEqual(result.compute(), 24.0)

    def generate_random_variable_name(self) -> str:
        """Generate a random variable name."""
        # Generate a random letter (a-z, A-Z)
        
        letter = random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        var_name = letter
        
        # Optionally add an underscore and a number
        if random.random() > 0.6:
            var_name += "_"
            var_name += str(random.randint(1, 999))

        return var_name

    def generate_random_expression(self, include_number=True, num_variables=None):
        """Generate a random valid expression without operators.
        
        Returns:
            tuple: (expression, variable_names, variable_positions)
            - expression: the generated expression string
            - variable_names: list of variable names in order of appearance
            - variable_positions: list of (start, end) positions for each variable
        """
        expression = ""
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
        
            variable_names.append(num_str)
            variable_positions.append((0, len(num_str)))

        # Step 2: Generate a random number of variables if not specified
        if num_variables is None:
            num_variables = random.randint(1, 10)
        
        var_names_set = set()

        # Step 3: Generate random variables
        for _ in range(num_variables):
            var_start = len(expression)
            
            var_name = self.generate_random_variable_name()
            
            while var_name in var_names_set:
                var_name = self.generate_random_variable_name()
            
            expression += var_name
            variable_names.append(var_name)
            var_names_set.add(var_name)
            variable_positions.append((var_start, var_start + len(var_name)))
        
        return expression, variable_names, variable_positions


    def test_parse_no_operators_random(self):
        """Test parse_no_operators with many randomly generated valid expressions."""
        for _ in range(5000):  
            # Generate a random expression
            expression, expected_vars, expected_positions = self.generate_random_expression()
            
            # Parse the expression
            result, pos_tracker, name_tracker = self.run_test(expression)

            # make sure the position tracking is correct            
            self.assertEqual(len(pos_tracker), len(expected_positions))
            for i, (pos, _) in enumerate(sorted(pos_tracker.items(), key=lambda x: x[0])):
                self.assertEqual(pos, expected_positions[i])


            # Verify that all variables were correctly parsed
            self.assertEqual(len(name_tracker), len(expected_vars))
            
            # Check that all expected variables are in the name tracker 
            for var_name in expected_vars:
                self.assertIn(var_name, name_tracker)
            

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
            
            
            # Parse the expression
            result, pos_tracker, name_tracker = self.run_test(num_str)
            
            # Verify results
            self.assertEqual(len(name_tracker), 1)  # No variables
            self.assertEqual(len(pos_tracker), 1)   # Only the number position
            self.assertIsInstance(result, ConstantVar)
            self.assertAlmostEqual(result.value, float(num_str))
            
            # Verify position tracking
            self.assertTrue(len(pos_tracker) == 1 and (0, len(num_str)) in pos_tracker)
    

    def test_parse_no_operators_variables_only(self):
        """Test parse_no_operators with expressions containing only variables (no numbers)."""
        for _ in range(5000):  
            # Generate a random expression with only variables
            expression, expected_vars, expected_positions = self.generate_random_expression(
                include_number=False, 
                num_variables=random.randint(1, 15)
            )
            
            # Parse the expression
            result, pos_tracker, name_tracker = self.run_test(expression)

            # make sure the position tracking is correct            
            self.assertEqual(len(pos_tracker), len(expected_positions))
            for i, (pos, _) in enumerate(sorted(pos_tracker.items(), key=lambda x: x[0])):
                self.assertEqual(pos, expected_positions[i])

            # Verify that all variables were correctly parsed
            self.assertEqual(len(name_tracker), len(expected_vars))
            
            # Check that all expected variables are in the name tracker
            for var_name in expected_vars:
                self.assertIn(var_name, name_tracker)
            
            # For multiple variables, should be a product
            self.assertIsInstance(result, Mult)
            


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    unittest.main()