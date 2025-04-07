"""
This file tests the exponentiation operator in combination with other operations
"""

import unittest
import random
import numpy as np

from tinyad.autoDiff.operators.binary_ops import Add, Mult, Sub, Div, Exp
from tinyad.autoDiff.var import ElementaryVar, ConstantVar
from tinyad.tests.operators.combined_ops_tests.base_test import BaseTest


class TestExponentialCombinations(BaseTest):
    """Test class for combined operations involving exponentiation."""
    
    def test_polynomial_with_exp(self):
        """Test polynomial expressions using Exp operator: ax^n + bx^m"""
        for _ in range(1000):
            # Create a variable with a random value
            x = ElementaryVar("x", random.uniform(0.5, 3.0))
            
            # Create random coefficients and exponents
            a = random.uniform(-3.0, 3.0)
            b = random.uniform(-3.0, 3.0)
            n = random.randint(0, 4)
            m = random.randint(0, 4)
            
            # Build the expression: ax^n + bx^m
            term1 = Mult(ConstantVar("a", a), Exp(x, ConstantVar("n", n)))
            term2 = Mult(ConstantVar("b", b), Exp(x, ConstantVar("m", m)))
            expr = Add(term1, term2)
            
            # Compute the result
            result = expr.compute()
            
            # Calculate expected result
            expected = a * (x.value ** n) + b * (x.value ** m)
            self.assertAlmostEqual(result, expected)
            
            # Test gradients
            expr.backward()
            
            # Expected gradient: d/dx(ax^n + bx^m) = a*n*x^(n-1) + b*m*x^(m-1)
            expected_grad = 0
            if n > 0:
                expected_grad += a * n * (x.value ** (n-1))
            if m > 0:
                expected_grad += b * m * (x.value ** (m-1))
                
            self.assertAlmostEqual(x.grad, expected_grad)
    
    def test_rational_with_exp(self):
        """Test rational expressions with exponentiation: x^n / (1 + x^m)"""
        for _ in range(1000):
            # Create a variable with a positive random value
            x = ElementaryVar("x", random.uniform(0.5, 3.0))
            
            # Create random exponents
            n = random.randint(1, 3)
            m = random.randint(1, 3)
            
            # Build the expression: x^n / (1 + x^m)
            numerator = Exp(x, ConstantVar("n", n))
            denominator = Add(ConstantVar("one", 1.0), Exp(x, ConstantVar("m", m)))
            expr = Div(numerator, denominator)
            
            # Compute the result
            result = expr.compute()
            
            # Calculate expected result
            expected = (x.value ** n) / (1 + x.value ** m)
            self.assertAlmostEqual(result, expected)
            
            # Test gradients
            expr.backward()
            
            # Expected gradient using quotient rule:
            # d/dx[f/g] = (g*df/dx - f*dg/dx)/g^2
            # where f = x^n and g = 1 + x^m
            
            f = x.value ** n
            g = 1 + x.value ** m
            df_dx = n * (x.value ** (n-1))
            dg_dx = m * (x.value ** (m-1))
            
            expected_grad = (g * df_dx - f * dg_dx) / (g ** 2)
            self.assertAlmostEqual(x.grad, expected_grad)
    
    def test_nested_exp(self):
        """Test nested exponentiation: (x^n)^m"""
        for _ in range(1000):
            # Create a variable with a positive random value
            x = ElementaryVar("x", random.uniform(0.5, 3.0))
            
            # Create random exponents
            n = random.randint(1, 3)
            m = random.randint(1, 3)
            
            # Build the nested expression: (x^n)^m
            inner_exp = Exp(x, ConstantVar("n", n))
            outer_exp = Exp(inner_exp, ConstantVar("m", m))
            
            # Compute the result
            result = outer_exp.compute()
            
            # Calculate expected result: (x^n)^m = x^(n*m)
            expected = x.value ** (n * m)
            self.assertAlmostEqual(result, expected)
            
            # Test gradients
            outer_exp.backward()
            
            # Expected gradient: d/dx[(x^n)^m] = m*n*x^(n*m-1)
            expected_grad = m * n * (x.value ** (n * m - 1))
            self.assertAlmostEqual(x.grad, expected_grad)
    
    def test_multivariate_exp(self):
        """Test expressions with multiple variables raised to powers: ax^n + by^m"""
        for _ in range(1000):
            # Create variables with random values
            x = ElementaryVar("x", random.uniform(0.5, 3.0))
            y = ElementaryVar("y", random.uniform(0.5, 3.0))
            
            # Create random coefficients and exponents
            a = random.uniform(-3.0, 3.0)
            b = random.uniform(-3.0, 3.0)
            n = random.randint(0, 4)
            m = random.randint(0, 4)
            
            # Build the expression: ax^n + by^m
            term1 = Mult(ConstantVar("a", a), Exp(x, ConstantVar("n", n)))
            term2 = Mult(ConstantVar("b", b), Exp(y, ConstantVar("m", m)))
            expr = Add(term1, term2)
            
            # Compute the result
            result = expr.compute()
            
            # Calculate expected result
            expected = a * (x.value ** n) + b * (y.value ** m)
            self.assertAlmostEqual(result, expected)
            
            # Test gradients
            expr.backward()
            
            # Expected gradients:
            # dx = a*n*x^(n-1)
            # dy = b*m*y^(m-1)
            expected_x_grad = 0
            if n > 0:
                expected_x_grad = a * n * (x.value ** (n-1))
                
            expected_y_grad = 0
            if m > 0:
                expected_y_grad = b * m * (y.value ** (m-1))
            
            self.assertAlmostEqual(x.grad, expected_x_grad)
            self.assertAlmostEqual(y.grad, expected_y_grad)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main() 