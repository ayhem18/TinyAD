"""
This is a file to test the correctness of the exponentiation operator
"""

import unittest
import random
import numpy as np

from tinyad.autoDiff.operators.binary_ops import Exp, Mult
from tinyad.autoDiff.var import ConstantVar, ElementaryVar


class TestExpForwardBasic(unittest.TestCase):
    """Test basic functionality of the Exp operator."""
    
    def test_init(self):
        """Test that Exp is initialized correctly."""
        x = ElementaryVar("x", 2.0)
        n = ConstantVar("n", 3.0)
        exp = Exp(x, n)
        
        self.assertEqual(exp.name, "x^n")
        self.assertIsNone(exp.value)
        self.assertIsNone(exp.grad)
        self.assertEqual(exp.children, [x, n])
        self.assertEqual(exp.left, x)
        self.assertEqual(exp.right, n)
        self.assertEqual(exp.exponent_value, 3.0)
    
        
    def test_compute(self):
        """Test the compute method with various random values."""
        # Test with 20 random pairs of values
        for _ in range(20):
            x_val = random.uniform(-10, 10)  # Positive values to avoid complex results

            n_val = 0
            while abs(n_val) < 1e-6:
                # keep generating a new exponent until it's not too close to 0
                n_val = random.uniform(-10, 5)   
            
            x = ElementaryVar("x", x_val)
            n = ConstantVar("n", n_val)
            exp = Exp(x, n)
            
            # Compute should return x_val ^ n_val
            expected = x_val ** n_val
            self.assertAlmostEqual(exp.compute(), expected)
            
            # Value should be cached after compute
            self.assertAlmostEqual(exp.value, expected)
    
    def test_forward(self):
        """Test the forward method creates a new Exp instance with forwarded operands."""
        x = ElementaryVar("x", 2.0)
        n = ConstantVar("n", 3.0)
        exp = Exp(x, n)
        
        forwarded = exp.forward()
        
        self.assertIsInstance(forwarded, Exp)
        self.assertIsNot(forwarded, exp)
        self.assertIs(forwarded.left, x)
        self.assertIs(forwarded.right, n)
    
    def test_call(self):
        """Test that calling an Exp instance invokes forward."""
        x = ElementaryVar("x", 2.0)
        n = ConstantVar("n", 3.0)
        exp = Exp(x, n)
        
        called = exp()
        
        self.assertIsInstance(called, Exp)
        self.assertIsNot(called, exp)
        self.assertIs(called.left, x)
        self.assertIs(called.right, n)
    
    def test_fail_with_zero_exponent(self):
        """Test that the exponent must be different from 0."""
        for _ in range(1000):
            x = ElementaryVar("x", random.uniform(0.1, 10))
            n = ConstantVar("n", 0)
            with self.assertRaises(ValueError):
                Exp(x, n)


class TestExpBackwardBasic(unittest.TestCase):
    """Test gradient computation for the Exp operator."""
    
    def test_simple_gradient(self):
        """Test gradient computation with integer exponents."""
        for base in range(1, 6):
            for power in range(1, 5):
                x = ElementaryVar("x", float(base))
                n = ConstantVar("n", float(power))
                exp = Exp(x, n)
                
                # Forward and backward pass
                exp.compute()
                exp.backward()
                
                # For y = x^n:
                # dy/dx = n * x^(n-1)
                expected_grad = power * (base ** (power - 1))
                
                self.assertAlmostEqual(x.grad, expected_grad)
                self.assertEqual(n.grad, 0)  # Constant always has zero gradient
    
    def test_different_upstream_gradients(self):
        """Test backpropagation with different upstream gradients."""
        for _ in range(100):
            x_val = random.uniform(-5, 5)
            n_val = 0
            while abs(n_val) < 1e-6:
                # keep generating a new exponent until it's not too close to 0
                n_val = random.uniform(-10, 5)   
            
            x = ElementaryVar("x", x_val)
            n = ConstantVar("n", n_val)
            
            exp = Exp(x, n)
            
            # Use different upstream gradients
            gradient1 = random.uniform(0.1, 10)
            gradient2 = random.uniform(0.1, 10)
            
            # Apply first gradient
            exp.compute()
            exp.backward(gradient1)
            
            x_grad1 = x.grad
            
            # Reset and apply second gradient
            x.grad = None
            exp.grad = None
            
            exp.backward(gradient2)
            
            # Check proportionality
            self.assertAlmostEqual(x_grad1 / x.grad, gradient1 / gradient2)
    
    def test_fractional_exponents(self):
        """Test gradient computation with fractional exponents."""
        for _ in range(1000):
            x_val = random.uniform(-10, 10)  # Positive values
            n_val = random.uniform(0.1, 2)  # Fractional exponents
            
            x = ElementaryVar("x", x_val)
            n = ConstantVar("n", n_val)
            exp = Exp(x, n)
            
            exp.compute()
            exp.backward()
            
            # Expected gradient: n * x^(n-1)
            expected_grad = n_val * (x_val ** (n_val - 1))
            self.assertAlmostEqual(x.grad, expected_grad)
    
    def test_complex_expression_tree(self):
        """Test gradients in a more complex expression tree with exponentiation."""
        for _ in range(1000):
            x_val = random.uniform(0.5, 3)
            
            x = ElementaryVar("x", x_val)
            
            # Create tree: x^2 * x^3 = x^5
            exp1 = Exp(x, ConstantVar("n1", 2))
            exp2 = Exp(x, ConstantVar("n2", 3))
            final = Mult(exp1, exp2)
            
            final.compute()
            final.backward()
            
            # Expected result: x^2 * x^3 = x^5
            expected_value = x_val ** 5
            self.assertAlmostEqual(final.value, expected_value)
            
            # Expected gradient: d/dx(x^5) = 5 * x^4
            expected_gradient = 5 * (x_val ** 4)
            self.assertAlmostEqual(x.grad, expected_gradient)
    
    def test_zero_base(self):
        """Test exponentiation with base 0."""
        for _ in range(1000):
            x = ElementaryVar("x", 0.0)
            n_val = 0
            while abs(n_val) < 1e-6:
                # keep generating a new exponent until it's not too close to 0
                n_val = random.uniform(-100, 100)   

            if n_val <= 0:
                with self.assertRaises(ValueError):
                    exp = Exp(x, ConstantVar("n", n_val))
                    exp.compute()
                continue
    
            # 0^anything = 0
            exp = Exp(x, ConstantVar("n", n_val))
            self.assertEqual(exp.compute(), 0.0)
            
            # Gradient should be 0, to prevent division by zero
            exp.backward()
            self.assertEqual(x.grad, 0.0)
            

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main() 