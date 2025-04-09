import random
import unittest

import numpy as np

from tinyad.parsing.parser import parse_number, parse_variable


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



if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    unittest.main()