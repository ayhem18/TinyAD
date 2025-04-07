import random
import unittest

import numpy as np

from tinyad.autoDiff.operators.binary_ops import Add, Mult, Sub, Div, Exp
from tinyad.autoDiff.var import ElementaryVar, ConstantVar
from tinyad.tests.operators.combined_ops_tests.base_test import BaseTest


class TestBasicCombinations(BaseTest):
    """Test class for combined operations of Add, Mult, Sub."""

    def test_mult_as_add(self):
        """Test the multiplication of two variables as an addition of two variables."""
        for _ in range(5000):            
            val = round(random.uniform(1, 10), 3)
            
            var1 = ElementaryVar("x", val)
            var2 = ElementaryVar("y", val)
            
            n = random.randint(2, 100)

            resAdd = ConstantVar("res", 0)
            
            for i in range(n):
                resAdd = Add(resAdd, var1)
            
            resMult = Mult(var2, ConstantVar("const", n))

            resAdd.compute()
            resMult.compute()

            self.assertEqual(var1.value, val)
            self.assertEqual(var2.value, val)

            self.assertAlmostEqual(resAdd.value, val * n)
            self.assertAlmostEqual(resMult.value, val * n)
            
            resAdd.backward()
            resMult.backward()

            self.assertEqual(var1.grad, n)
            self.assertEqual(var2.grad, n)

    
    def test_mult_as_sub(self):
        """Test the multiplication of two variables as a subtraction of two variables."""
        for _ in range(5000):
            val = round(random.uniform(1, 10), 3)
            
            var1 = ElementaryVar("x", val)
            var2 = ElementaryVar("y", val)

            n = random.randint(2, 100)

            resSub = ConstantVar("res", 0)
            for i in range(n):
                resSub = Sub(resSub, var1)
            
            resMult = Mult(var2, ConstantVar("const", -n))

            resSub.compute()
            resMult.compute()

            self.assertEqual(var1.value, val)
            self.assertEqual(var2.value, val)

            self.assertAlmostEqual(resSub.value, -val * n)
            self.assertAlmostEqual(resMult.value, -val * n)

            resSub.backward()
            resMult.backward()

            self.assertEqual(var1.grad, -n)
            self.assertEqual(var2.grad, -n)


    def test_same_powers_divided(self):
        """
        Test division of the same variable raised to the same power.
        Tests the expression x^n / x^n = 1 and verifies gradient is zero.
        """
        for _ in range(1000):
            # Avoid values too close to zero to prevent numerical instability
            val = round(random.uniform(0.5, 10.0), 3)
            var = ElementaryVar("x", val)
            
            # Choose a random exponent
            n = random.randint(1, 5)
            
            # Create x^n for numerator
            numerator = var
            for _ in range(n-1):
                numerator = Mult(numerator, var)
            
            # Create x^n for denominator
            denominator = var
            for _ in range(n-1):
                denominator = Mult(denominator, var)
            
            # Create division expression: x^n / x^n
            division = Div(numerator, denominator)
            
            # Compute result - should be 1
            result = division.compute()
            self.assertAlmostEqual(result, 1.0)
            
            # Perform backward pass - gradient should be 0
            division.backward()
            self.assertAlmostEqual(var.grad, 0.0)  # Use places to handle potential floating point issues


    def test_diff_powers_divided(self):
        """
        Test division of the same variable raised to different powers.
        Tests the expression x^n1 / x^n2 = x^(n1-n2) and verifies gradient.
        """
        for _ in range(1000):
            # Avoid values too close to zero to prevent numerical instability
            val = round(random.uniform(0.5, 10.0), 3)
            var = ElementaryVar("x", val)
            
            # Choose two different exponents
            n1 = random.randint(2, 6)
            n2 = random.randint(1, n1-1)  # Ensure n1 > n2 to avoid negative powers
            
            # Create x^n1 for numerator
            numerator = var
            for _ in range(n1-1):
                numerator = Mult(numerator, var)
            
            # Create x^n2 for denominator
            denominator = var
            for _ in range(n2-1):
                denominator = Mult(denominator, var)
            
            # Create division expression: x^n1 / x^n2
            division = Div(numerator, denominator)
            
            # Compute result - should be x^(n1-n2)
            result = division.compute()
            expected_result = val ** (n1 - n2)
            self.assertAlmostEqual(result, expected_result)
            
            # Perform backward pass
            division.backward()
            
            # Calculate expected gradient: d/dx(x^(n1-n2)) = (n1-n2) * x^(n1-n2-1)
            power_diff = n1 - n2
            expected_gradient = power_diff * (val ** (power_diff - 1))
            
            self.assertAlmostEqual(var.grad, expected_gradient)


    def test_exp_as_mult(self):
        """
        Test the exponentiation of a variable as a repeated multiplication.
        Tests x^n = x * x * ... * x (n times) and verifies both values and gradients match.
        """
        for _ in range(1000):
            # Avoid values too close to zero to prevent numerical instability
            val = round(random.uniform(0.5, 5.0), 3)
            var1 = ElementaryVar("x", val)
            var2 = ElementaryVar("y", val)  # Same value but separate variable for Exp
            
            # Choose a random integer exponent
            n = random.randint(1, 5)
            
            # Create x^n using repeated multiplication
            mult_result = var1
            for _ in range(n-1):
                mult_result = Mult(mult_result, var1)
            
            # Create x^n using Exp operator
            exp_result = Exp(var2, ConstantVar("n", n))
            
            # Compute both expressions
            mult_value = mult_result.compute()
            exp_value = exp_result.compute()
            
            # Values should match
            self.assertAlmostEqual(mult_value, val**n)
            self.assertAlmostEqual(exp_value, val**n)
            self.assertAlmostEqual(mult_value, exp_value)
            
            # Test gradients
            mult_result.backward()
            exp_result.backward()
            
            # Expected gradient for x^n: n * x^(n-1)
            expected_gradient = n * (val**(n-1))
            
            # Gradients should match
            self.assertAlmostEqual(var1.grad, expected_gradient)
            self.assertAlmostEqual(var2.grad, expected_gradient)



if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main()
