"""
This is a file to test the correctness of the absolute value operator
"""

import unittest
import random
import numpy as np

from tinyad.autoDiff.operators.unary_ops import AbsVal
from tinyad.autoDiff.var import ConstantVar, ElementaryVar


class TestAbsValForwardBasic(unittest.TestCase):
    """Test basic functionality of the AbsVal operator."""
    
    def test_init(self):
        """Test that AbsVal is initialized correctly."""
        x = ElementaryVar("x", -2.0)
        abs_op = AbsVal(x)
        
        self.assertEqual(abs_op.name, "|x|")
        self.assertIsNone(abs_op.value)
        self.assertIsNone(abs_op.grad)
        self.assertEqual(abs_op.children, [x])
        self.assertEqual(abs_op.operand, x)
    
    def test_compute(self):
        """Test the compute method with various random values."""
        for _ in range(1000):
            x_val = random.uniform(-100, 100)
            x = ElementaryVar("x", x_val)
            abs_op = AbsVal(x)
            
            # Compute should return |x_val|
            expected = abs(x_val)
            self.assertEqual(abs_op.compute(), expected)
            
            # Value should be cached after compute
            self.assertEqual(abs_op.value, expected)
    
    def test_forward(self):
        """Test the forward method creates a new AbsVal instance with forwarded operand."""
        x = ElementaryVar("x", -2.0)
        abs_op = AbsVal(x)
        
        forwarded = abs_op.forward()
        
        self.assertIsInstance(forwarded, AbsVal)
        self.assertIsNot(forwarded, abs_op)
        self.assertIs(forwarded.operand, x)
    
    def test_call(self):
        """Test that calling an AbsVal instance invokes forward."""
        x = ElementaryVar("x", -2.0)
        abs_op = AbsVal(x)
        
        called = abs_op()
        
        self.assertIsInstance(called, AbsVal)
        self.assertIsNot(called, abs_op)
        self.assertIs(called.operand, x)
    
    def test_nested_compute(self):
        """Test computation with nested AbsVal operators."""
        for _ in range(1000):
            x_val = random.uniform(-100, 100)
            x = ElementaryVar("x", x_val)
            
            # |x|
            abs1 = AbsVal(x)
            expected1 = abs(x_val)
            self.assertEqual(abs1.compute(), expected1)
            
            # ||x|| = |x|
            abs2 = AbsVal(abs1)
            self.assertEqual(abs2.compute(), expected1)  # Always positive
    
    def test_abs_with_constants(self):
        """Test taking absolute value of constants."""
        for _ in range(1000):
            const_val = random.uniform(-100, 100)
            const = ConstantVar("c", const_val)
            abs_op = AbsVal(const)
            
            self.assertEqual(abs_op.compute(), abs(const_val))
    
    def test_zero_value(self):
        """Test absolute value with zero."""
        x = ElementaryVar("x", 0.0)
        abs_op = AbsVal(x)
        
        self.assertEqual(abs_op.compute(), 0.0)


class TestAbsValBackwardBasic(unittest.TestCase):
    """Test gradient computation for the AbsVal operator."""
    
    def test_positive_gradient(self):
        """Test gradient computation for positive values."""
        for _ in range(1000):
            x_val = random.uniform(0.1, 100)  # Positive values
            x = ElementaryVar("x", x_val)
            abs_op = AbsVal(x)
            
            abs_op.compute()
            abs_op.backward()
            
            # For y = |x| where x > 0: dy/dx = 1
            self.assertEqual(x.grad, 1)
    
    def test_negative_gradient(self):
        """Test gradient computation for negative values."""
        for _ in range(1000):
            x_val = random.uniform(-100, -0.1)  # Negative values
            x = ElementaryVar("x", x_val)
            abs_op = AbsVal(x)
            
            abs_op.compute()
            abs_op.backward()
            
            # For y = |x| where x < 0: dy/dx = -1
            self.assertEqual(x.grad, -1)
    
    def test_zero_gradient(self):
        """Test gradient computation at x = 0 (should be 0 as specified)."""
        x = ElementaryVar("x", 0.0)
        abs_op = AbsVal(x)
        
        abs_op.compute()
        abs_op.backward()
        
        # For y = |x| where x = 0: dy/dx = 0 (a design choice)
        self.assertEqual(x.grad, 0)
    
    def test_nested_gradients(self):
        """Test gradient computation with nested AbsVal operators."""
        for _ in range(1000):
            x_val = random.uniform(-10, 10)
            x = ElementaryVar("x", x_val)
            
            # |x|
            abs1 = AbsVal(x)
            
            # ||x||
            abs2 = AbsVal(abs1)
            
            abs2.compute()
            abs2.backward()
            
            # For ||x|| = |x|, since |x| is always non-negative:
            # d|x|/dx = sign(x)
            # d||x||/d|x| = 1
            # So d||x||/dx = sign(x)
            expected_grad = np.sign(x_val)
            self.assertEqual(x.grad, expected_grad)
    
    def test_different_upstream_gradients(self):
        """Test backpropagation with different upstream gradients."""
        for _ in range(1000):
            x = ElementaryVar("x", random.uniform(1, 10))  # Use positive for simplicity
            abs_op = AbsVal(x)
            
            # Use different upstream gradients
            gradient1 = random.uniform(0.1, 10)
            gradient2 = random.uniform(0.1, 10)
            
            # Apply first gradient
            abs_op.compute()
            abs_op.backward(gradient1)
            x_grad1 = x.grad
            
            # Reset and apply second gradient
            x.grad = None
            abs_op.grad = None
            abs_op.backward(gradient2)
            
            # Check proportionality
            self.assertAlmostEqual(x_grad1 / x.grad, gradient1 / gradient2)
    
    def test_numerical_stability(self):
        """Test AbsVal with very large and very small numbers."""
        # Test with very large negative number
        x_large_neg = ElementaryVar("x_large_neg", -1e15)
        abs_large_neg = AbsVal(x_large_neg)
        self.assertEqual(abs_large_neg.compute(), 1e15)
        
        # Test with very small number
        x_small = ElementaryVar("x_small", 1e-15)
        abs_small = AbsVal(x_small)
        self.assertEqual(abs_small.compute(), 1e-15)
        
        # Test with very small negative number
        x_small_neg = ElementaryVar("x_small_neg", -1e-15)
        abs_small_neg = AbsVal(x_small_neg)
        self.assertEqual(abs_small_neg.compute(), 1e-15)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main() 