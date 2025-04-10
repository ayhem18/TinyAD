"""
This is a file to test the correctness of the negation operator
"""

import unittest
import random
import numpy as np

from tinyad.autoDiff.operators.unary_ops import Neg
from tinyad.autoDiff.var import ConstantVar, ElementaryVar


class TestNegForwardBasic(unittest.TestCase):
    """Test basic functionality of the Neg operator."""
    
    def test_init(self):
        """Test that Neg is initialized correctly."""
        x = ElementaryVar("x", 2.0)
        neg = Neg(x)
        
        self.assertEqual(neg.name, "-x")
        self.assertIsNone(neg.value)
        self.assertIsNone(neg.grad)
        self.assertEqual(neg.children, [x])
        self.assertEqual(neg.operand, x)
    
    def test_compute(self):
        """Test the compute method with various random values."""
        for _ in range(1000):
            x_val = random.uniform(-100, 100)
            x = ElementaryVar("x", x_val)
            neg = Neg(x)
            
            # Compute should return -x_val
            expected = -x_val
            self.assertEqual(neg.compute(), expected)
            
            # Value should be cached after compute
            self.assertEqual(neg.value, expected)
    
    def test_forward(self):
        """Test the forward method creates a new Neg instance with forwarded operand."""
        x = ElementaryVar("x", 2.0)
        neg = Neg(x)
        
        forwarded = neg.forward()
        
        self.assertIsInstance(forwarded, Neg)
        self.assertIsNot(forwarded, neg)
        self.assertIs(forwarded.operand, x)
    
    def test_call(self):
        """Test that calling a Neg instance invokes forward."""
        x = ElementaryVar("x", 2.0)
        neg = Neg(x)
        
        called = neg()
        
        self.assertIsInstance(called, Neg)
        self.assertIsNot(called, neg)
        self.assertIs(called.operand, x)
    
    def test_nested_compute(self):
        """Test computation with nested Neg operators."""
        for _ in range(1000):
            x_val = random.uniform(-100, 100)
            x = ElementaryVar("x", x_val)
            
            # -x
            neg1 = Neg(x)
            self.assertEqual(neg1.compute(), -x_val)
            
            # -(-x) = x
            neg2 = Neg(neg1)
            self.assertEqual(neg2.compute(), x_val)
            
            # -(-(-x)) = -x
            neg3 = Neg(neg2)
            self.assertEqual(neg3.compute(), -x_val)
    
    def test_neg_with_constants(self):
        """Test negating constants."""
        for _ in range(1000):
            const_val = random.uniform(-100, 100)
            const = ConstantVar("c", const_val)
            neg = Neg(const)
            
            self.assertEqual(neg.compute(), -const_val)


class TestNegBackwardBasic(unittest.TestCase):
    """Test gradient computation for the Neg operator."""
    
    def test_simple_gradient(self):
        """Test gradient computation for a simple case."""
        for _ in range(1000):
            x_val = random.uniform(-100, 100)
            x = ElementaryVar("x", x_val)
            neg = Neg(x)
            
            neg.compute()
            neg.backward()
            
            # For y = -x: dy/dx = -1
            self.assertEqual(x.grad, -1)
    
    def test_nested_gradients(self):
        """Test gradient computation with nested Neg operators."""
        for _ in range(1000):
            x_val = random.uniform(-100, 100)
            x = ElementaryVar("x", x_val)
            
            # -x
            neg1 = Neg(x)
            neg1.compute()
            neg1.backward()
            self.assertEqual(x.grad, -1)
            
            # Reset
            x.grad = None
            
            # -(-x) = x
            neg2 = Neg(neg1)
            neg2.compute()
            neg2.backward()
            self.assertEqual(x.grad, 1)  # -1 * -1 = 1
            
            # Reset
            x.grad = None
            
            # -(-(-x)) = -x
            neg3 = Neg(neg2)
            neg3.compute()
            neg3.backward()
            self.assertEqual(x.grad, -1)  # -1 * -1 * -1 = -1
    

    def test_different_upstream_gradients(self):
        """Test backpropagation with different upstream gradients."""
        for _ in range(1000):
            x = ElementaryVar("x", random.uniform(-10, 10))
            neg = Neg(x)
            
            # Use different upstream gradients
            gradient1 = random.uniform(0.1, 10)
            gradient2 = random.uniform(0.1, 10)
            
            # Apply first gradient
            neg.compute()
            neg.backward(gradient1)
            x_grad1 = x.grad
            
            # Reset and apply second gradient
            x.grad = None
            neg.grad = None
            neg.backward(gradient2)
            
            # Check proportionality but with negative sign
            self.assertAlmostEqual(x_grad1 / x.grad, gradient1 / gradient2)
    
    def test_numerical_stability(self):
        """Test Neg with very large and very small numbers."""
        # Test with very large numbers
        x_large = ElementaryVar("x_large", 1e15)
        neg_large = Neg(x_large)
        self.assertEqual(neg_large.compute(), -1e15)
        
        # Test with very small numbers
        x_small = ElementaryVar("x_small", 1e-15)
        neg_small = Neg(x_small)
        self.assertEqual(neg_small.compute(), -1e-15)

    def test_multiple_negations(self):
        """
        Test applying the negation operator multiple times.
        For n negations:
        - If n is even: the result should be x and gradient should be 1
        - If n is odd: the result should be -x and gradient should be -1
        """
        for _ in range(100):  # Run multiple test cases
            # 1. Generate a random number of negations between 2 and 50
            n_negations = random.randint(2, 50)
            x_val = random.uniform(-10, 10)
            x = ElementaryVar("x", x_val)
            
            # 2. Apply negation n times
            result = x
            for _ in range(n_negations):
                result = Neg(result)
            
            # 3. Compute the result
            computed_result = result.compute()
            
            # 4. Check forward computation based on parity of n_negations
            if n_negations % 2 == 0:  # Even number of negations
                # n even: -(-(...(-x))) = x
                expected_result = x_val
            else:  # Odd number of negations
                # n odd: -(-(...(-(-x)))) = -x
                expected_result = -x_val
                
            self.assertEqual(computed_result, expected_result)
            
            # 5. Perform backward pass
            result.backward()

            # the expression below evaluates to 1 if n_negations is even, and -1 otherwise            
            expected_gradient = 2 * int(n_negations % 2 == 0) - 1
                
            self.assertEqual(x.grad, expected_gradient, 
                             f"Gradient incorrect for {n_negations} negations. Expected {expected_gradient}, got {x.grad}")

    def test_zero_value_negations(self):
        """
        Test the negation operator with a variable of value 0.
        The result should always be 0 for any number of negations,
        but the gradient should follow the same pattern as non-zero values.
        """
        # 1. Test single negation of zero
        x_zero = ElementaryVar("x_zero", 0.0)
        neg_zero = Neg(x_zero)
        
        # Negation of zero should be zero
        self.assertEqual(neg_zero.compute(), 0.0)
        
        # Gradient should still be -1
        neg_zero.backward()
        self.assertEqual(x_zero.grad, -1)
        
        # 2. Test multiple negations of zero
        for _ in range(1000):
            # Generate a random number of negations
            n_negations = random.randint(1, 30)
            x = ElementaryVar("x", 0.0)
            
            # Apply n negations
            result = x
            for _ in range(n_negations):
                result = Neg(result)
            
            # The result should always be 0
            self.assertEqual(result.compute(), 0.0)
            
            # Perform backward pass
            result.backward()

            # the expression below evaluates to 1 if n_negations is even, and -1 otherwise            
            expected_gradient = 2 * int(n_negations % 2 == 0) - 1

            self.assertEqual(x.grad, expected_gradient, 
                            f"Gradient incorrect for {n_negations} negations of zero. Expected {expected_gradient}, got {x.grad}")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main() 