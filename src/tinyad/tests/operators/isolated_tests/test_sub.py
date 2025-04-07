"""
This is a file to test the correctness of the subtraction operator
"""


import unittest
import random

from tinyad.autoDiff.operators.binary_ops import Sub
from tinyad.autoDiff.var import ConstantVar, ElementaryVar


class TestSubForwardBasic(unittest.TestCase):  
    """Test basic functionality of the Sub operator."""
    
    def test_init(self):
        """Test that Sub is initialized correctly."""
        x = ElementaryVar("x", 2.0)
        y = ElementaryVar("y", 3.0)
        sub = Sub(x, y)
        
        self.assertEqual(sub.name, "-")
        self.assertIsNone(sub.value)
        self.assertIsNone(sub.grad)
        self.assertEqual(sub.children, [x, y])
        self.assertEqual(sub.left, x)
        self.assertEqual(sub.right, y)
    
    def test_compute(self):
        """Test the compute method with various random values."""
        # Test with 20 random pairs of values
        for _ in range(20):
            x_val = random.uniform(-100, 100)
            y_val = random.uniform(-100, 100)
            
            x = ElementaryVar("x", x_val)
            y = ElementaryVar("y", y_val)
            sub = Sub(x, y)
            
            # Compute should return x_val - y_val
            expected = x_val - y_val
            self.assertEqual(sub.compute(), expected)
            
            # Value should be cached after compute
            self.assertEqual(sub.value, expected)
    
    def test_forward(self):
        """Test the forward method creates a new Sub instance with forwarded operands."""
        x = ElementaryVar("x", 2.0)
        y = ElementaryVar("y", 3.0)
        sub = Sub(x, y)
        
        # Since ElementaryVar.forward() returns self, the forwarded Sub should 
        # have the same operands but be a different instance
        forwarded = sub.forward()
        
        self.assertIsInstance(forwarded, Sub)
        self.assertIsNot(forwarded, sub)
        self.assertIs(forwarded.left, x)
        self.assertIs(forwarded.right, y)
    
    def test_call(self):
        """Test that calling a Sub instance invokes forward."""
        x = ElementaryVar("x", 2.0)
        y = ElementaryVar("y", 3.0)
        sub = Sub(x, y)
        
        # Called should be equivalent to forward
        called = sub()
        
        self.assertIsInstance(called, Sub)
        self.assertIsNot(called, sub)
        self.assertIs(called.left, x)
        self.assertIs(called.right, y)
    
    def test_nested_compute(self):
        """Test computation with nested Sub operators."""
        for _ in range(50):
            
            result_val = 0
            result = Sub(ConstantVar("c", 0), ConstantVar("c", 0))

            for _ in range(10):
                val = random.uniform(-100, 100)
                # make sure the sub works with the ConstantVar variable 
                result = Sub(result, ConstantVar("c", val))
                
                result_val -= val
                self.assertEqual(result.compute(), result_val)

                # make sure it works with ElementaryVar variables
                x = ElementaryVar("x", val)
                result = Sub(result, x)
                result_val -= val
                self.assertEqual(result.compute(), result_val)


    def test_sub_with_constants(self):
        """Test subtracting constants from variables."""
        for _ in range(10):
            x_val = random.uniform(-100, 100)
            const_val = random.uniform(-100, 100)
            
            x = ElementaryVar("x", x_val)
            const = ConstantVar("c", const_val)
            
            # Test x - const
            sub1 = Sub(x, const)
            self.assertEqual(sub1.compute(), x_val - const_val)
            
            # Test const - x
            sub2 = Sub(const, x)
            self.assertEqual(sub2.compute(), const_val - x_val)
            
            # Test gradient - should only flow to the ElementaryVar
            # For x - const: gradient to x is 1, to const is 0
            sub1.backward()
            self.assertEqual(x.grad, 1.0)
            self.assertEqual(const.grad, 0)
            
            # Reset gradients
            x.grad = None
            const.grad = None
            
            # For const - x: gradient to const is 0, to x is -1
            sub2.backward()
            self.assertEqual(const.grad, 0)
            self.assertEqual(x.grad, -1.0)


class TestSubBackwardBasic(unittest.TestCase):
    """Test gradient computation for the Sub operator."""
    
    def test_simple_gradient(self):
        """Test gradient computation with two ElementaryVars."""
        for i in range(1, 11):
            for j in range(1, 11):
                x = ElementaryVar("x", i)
                y = ElementaryVar("y", j)
                sub = Sub(x, y)
                
                # Forward and backward pass
                sub.compute()  # Ensure value is computed
                sub.backward()  # Default gradient of 1.0
                
                # For z = x - y:
                # dz/dx = 1, dz/dy = -1 (regardless of the values of x and y)
                self.assertEqual(sub.grad, 1.0)
                self.assertEqual(x.grad, 1.0)
                self.assertEqual(y.grad, -1.0)  # Right operand gets negative gradient
                
                # Test with a different upstream gradient
                x.grad = None
                y.grad = None
                sub.grad = None
                
                value = random.random()
                sub.backward(value)
                
                self.assertEqual(sub.grad, value)
                self.assertEqual(x.grad, value)
                self.assertEqual(y.grad, -value)  # Negative gradient scaled by upstream value

    def test_gradient_involved(self):
        """
        Test gradient computation in a more complex scenario with multiple Sub operations.
        """
        for _ in range(10):
            # 1. Generate n from 3 to 10
            n = random.randint(3, 10)
            
            # 2. Create n ElementaryVar objects
            variables = [ElementaryVar(f"var{i}", random.uniform(-10, 10)) for i in range(n)]
            
            counts = [0] * n  # Times a variable appears on the left (positive gradient)
            
            # 3. Create an initial Sub using the first two variables
            result = Sub(ConstantVar("c", 0), variables[1])
            counts[1] += 1
            
            # 4. Inner loop - subtract more variables from the expression
            for _ in range(50):
                # 5. Choose a random variable to subtract
                idx = random.randint(0, n - 1)
                
                # Update the expression: result = result - variables[idx]
                result = Sub(result, variables[idx])
                counts[idx] += 1
            
            # Compute the result
            result.compute()
            
            # Perform backward pass with gradient of 1.0
            result.backward()
            
            # 6. Check that each variable's gradient matches its count but to the negative sign
            for i in range(n):
                if counts[i] == 0:
                    self.assertIsNone(variables[i].grad)
                else:
                    self.assertEqual(variables[i].grad, -counts[i],
                                    f"Variable {i} gradient {variables[i].grad} does not match expected {-counts[i]}")


    def test_different_upstream_gradients(self):
        """Test backpropagation with different upstream gradients."""
        for _ in range(10):
            x = ElementaryVar("x", random.uniform(-10, 10))
            y = ElementaryVar("y", random.uniform(-10, 10))
            
            result = Sub(x, y)
            
            # Use different upstream gradients
            gradient1 = random.uniform(0.1, 10)
            gradient2 = random.uniform(0.1, 10)
            
            # Apply first gradient
            result.compute()
            result.backward(gradient1)
            
            x_grad1 = x.grad
            y_grad1 = y.grad
            
            # Reset and apply second gradient
            x.grad = None
            y.grad = None
            result.grad = None
            
            result.backward(gradient2)
            
            # Check proportionality
            self.assertAlmostEqual(x_grad1 / x.grad, gradient1 / gradient2)
            self.assertAlmostEqual(y_grad1 / y.grad, gradient1 / gradient2)


    def test_numerical_stability(self):
        """Test Sub with very large and very small numbers."""
        # Test with very large numbers
        x_large = ElementaryVar("x_large", 1e15)
        y_large = ElementaryVar("y_large", 1e15)
        sub_large = Sub(x_large, y_large)
        self.assertEqual(sub_large.compute(), 0)
        
        # Test with very small numbers
        x_small = ElementaryVar("x_small", 1e-15)
        y_small = ElementaryVar("y_small", 1e-15)
        sub_small = Sub(x_small, y_small)
        self.assertEqual(sub_small.compute(), 0)
        
        # Test with mix of large and small
        sub_mixed = Sub(x_large, x_small)

        self.assertEqual(sub_mixed.compute(), 1e15 - 1e-15)

    def test_complex_expression_tree(self):
        """Test gradients in a more complex expression tree with branches."""
        x = ElementaryVar("x", 2.0)
        y = ElementaryVar("y", 3.0)
        z = ElementaryVar("z", 4.0)
        
        # Create tree: 
        #       final
        #      /     \
        #   sub1     sub2
        #   /  \     /  \
        #  x    y   y    z
        
        sub1 = Sub(x, y)
        sub2 = Sub(y, z)
        final = Sub(sub1, sub2)
        
        final.compute()
        final.backward()
        
        # For final = sub1 - sub2 = (x - y) - (y - z) = x - 2y + z
        # dx/dfinal = 1
        # dy/dfinal = -2
        # dz/dfinal = 1
        self.assertEqual(x.grad, 1.0)
        self.assertEqual(y.grad, -2.0)
        self.assertEqual(z.grad, 1.0)


if __name__ == '__main__':
    unittest.main()
