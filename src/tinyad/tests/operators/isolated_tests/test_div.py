"""
This is a file to test the correctness of the division operator
"""

import unittest
import random
import numpy as np

from tinyad.autoDiff.operators.binary_ops import Div
from tinyad.autoDiff.var import ConstantVar, ElementaryVar


class TestDivForwardBasic(unittest.TestCase):
    """Test basic functionality of the Div operator."""
    
    def test_init(self):
        """Test that Div is initialized correctly."""
        x = ElementaryVar("x", 6.0)
        y = ElementaryVar("y", 3.0)
        div = Div(x, y)
        
        self.assertEqual(div.name, "/")
        self.assertIsNone(div.value)
        self.assertIsNone(div.grad)
        self.assertEqual(div.children, [x, y])
        self.assertEqual(div.left, x)
        self.assertEqual(div.right, y)
    
    def test_compute(self):
        """Test the compute method with various random values."""
        # Test with 20 random pairs of values
        for _ in range(20):
            x_val = random.uniform(-100, 100)
            y_val = random.uniform(1, 100)  # Avoid division by zero
            
            x = ElementaryVar("x", x_val)
            y = ElementaryVar("y", y_val)
            div = Div(x, y)
            
            # Compute should return x_val / y_val
            expected = x_val / y_val
            self.assertAlmostEqual(div.compute(), expected)
            
            # Value should be cached after compute
            self.assertAlmostEqual(div.value, expected)
    
    def test_forward(self):
        """Test the forward method creates a new Div instance with forwarded operands."""
        x = ElementaryVar("x", 6.0)
        y = ElementaryVar("y", 3.0)
        div = Div(x, y)
        
        # Since ElementaryVar.forward() returns self, the forwarded Div should 
        # have the same operands but be a different instance
        forwarded = div.forward()
        
        self.assertIsInstance(forwarded, Div)
        self.assertIsNot(forwarded, div)
        self.assertIs(forwarded.left, x)
        self.assertIs(forwarded.right, y)
    
    def test_call(self):
        """Test that calling a Div instance invokes forward."""
        x = ElementaryVar("x", 6.0)
        y = ElementaryVar("y", 3.0)
        div = Div(x, y)
        
        # Called should be equivalent to forward
        called = div()
        
        self.assertIsInstance(called, Div)
        self.assertIsNot(called, div)
        self.assertIs(called.left, x)
        self.assertIs(called.right, y)
    
    def test_nested_compute(self):
        """Test computation with nested Div operators."""
        for _ in range(20):
            # Start with a value between 1000-10000 to avoid underflow
            result_val = random.uniform(1000, 10000)
            result = ElementaryVar("start", result_val)
            
            # Divide by multiple values
            for _ in range(5):  # Fewer iterations to avoid tiny values
                # Use values between 1.1 and 5.0 to avoid division by very small numbers
                val = random.uniform(1.1, 5.0)  
                # Test with ConstantVar
                const = ConstantVar("c", val)
                result = Div(result, const)
                
                result_val /= val
                self.assertAlmostEqual(result.compute(), result_val)
                
                # Test with ElementaryVar
                x = ElementaryVar("x", val)
                result = Div(result, x)
                
                result_val /= val
                self.assertAlmostEqual(result.compute(), result_val)

    def test_div_with_constants(self):
        """Test dividing variables with constants."""
        for _ in range(10):
            x_val = random.uniform(-100, 100)
            const_val = random.uniform(1, 10)  # Avoid division by zero
            
            x = ElementaryVar("x", x_val)
            const = ConstantVar("c", const_val)
            
            # Test x / const
            div1 = Div(x, const)
            self.assertAlmostEqual(div1.compute(), x_val / const_val)
            
            # Test const / x (only when x is not near zero)
            if abs(x_val) > 0.001:
                div2 = Div(const, x)
                self.assertAlmostEqual(div2.compute(), const_val / x_val)


class TestDivBackwardBasic(unittest.TestCase):
    """Test gradient computation for the Div operator."""
    
    def test_simple_gradient(self):
        """Test gradient computation with two ElementaryVars."""
        for i in range(1, 11):
            for j in range(1, 11):
                x = ElementaryVar("x", i)
                y = ElementaryVar("y", j)
                div = Div(x, y)
                
                # Forward and backward pass
                div.compute()  # Ensure value is computed
                div.backward()  # Default gradient of 1.0
                
                # For z = x / y:
                # dz/dx = 1/y, dz/dy = -x/y²
                self.assertEqual(div.grad, 1.0)
                self.assertAlmostEqual(x.grad, 1.0 / j)
                self.assertAlmostEqual(y.grad, -i / (j * j))
                
                # Test with a different upstream gradient
                x.grad = None
                y.grad = None
                div.grad = None
                
                value = random.random()
                div.backward(value)
                
                self.assertEqual(div.grad, value)
                self.assertAlmostEqual(x.grad, value / j)
                self.assertAlmostEqual(y.grad, -value * i / (j * j))

    def test_gradient_accumulation(self):
        """Test that gradients accumulate properly in shared variables."""
        x = ElementaryVar("x", 6.0)
        y = ElementaryVar("y", 2.0)
        z = ElementaryVar("z", 3.0)
        
        # Create two Div operations that share variable x
        div1 = Div(x, y)  # x/y = 6/2 = 3
        div2 = Div(x, z)  # x/z = 6/3 = 2
        
        # Compute both and backpropagate
        div1.compute()
        div2.compute()
        
        div1.backward()
        div2.backward()
        
        # x should receive gradient from both operations: 1/y + 1/z = 1/2 + 1/3 = 5/6
        self.assertAlmostEqual(x.grad, 1/y.value + 1/z.value)
        # y's gradient is -x/y² = -6/4 = -1.5
        self.assertAlmostEqual(y.grad, -x.value/(y.value**2))
        # z's gradient is -x/z² = -6/9 = -2/3
        self.assertAlmostEqual(z.grad, -x.value/(z.value**2))

    def test_different_upstream_gradients(self):
        """Test backpropagation with different upstream gradients."""
        for _ in range(10):
            x_val = random.uniform(1, 10)
            y_val = random.uniform(1, 10)  # Avoid zeros and tiny values
            
            x = ElementaryVar("x", x_val)
            y = ElementaryVar("y", y_val)
            
            div = Div(x, y)
            
            # Use different upstream gradients
            gradient1 = random.uniform(0.1, 10)
            gradient2 = random.uniform(0.1, 10)
            
            # Apply first gradient
            div.compute()
            div.backward(gradient1)
            
            x_grad1 = x.grad
            y_grad1 = y.grad
            
            # Reset and apply second gradient
            x.grad = None
            y.grad = None
            div.grad = None
            
            div.backward(gradient2)
            
            # Check proportionality
            self.assertAlmostEqual(x_grad1 / x.grad, gradient1 / gradient2)
            self.assertAlmostEqual(y_grad1 / y.grad, gradient1 / gradient2)


    def test_numerical_stability(self):
        """Test Div with various numerical edge cases."""
        # Large numerator, normal denominator
        x_large = ElementaryVar("x_large", 1e10)
        y_normal = ElementaryVar("y_normal", 2.0)
        div1 = Div(x_large, y_normal)
        self.assertAlmostEqual(div1.compute(), 5e9)
        
        # Normal numerator, large denominator
        x_normal = ElementaryVar("x_normal", 5.0)
        y_large = ElementaryVar("y_large", 1e10)
        div2 = Div(x_normal, y_large)
        self.assertAlmostEqual(div2.compute(), 5e-10)
        
        # Very small numerator (but not zero)
        x_small = ElementaryVar("x_small", 1e-10)
        y_normal = ElementaryVar("y_normal", 2.0) 
        div3 = Div(x_small, y_normal)
        self.assertAlmostEqual(div3.compute(), 5e-11)
        
        # Very small denominator (but not zero) - handle carefully to avoid div by zero issues
        x_normal = ElementaryVar("x_normal", 5.0)
        y_small = ElementaryVar("y_small", 1e-5)  # Small but not too small
        div4 = Div(x_normal, y_small)
        self.assertAlmostEqual(div4.compute(), 5e5)


    def test_complex_expression_tree(self):
        """Test gradients in a more complex expression tree with divisions."""
        for _ in range(10):
            # Use more stable random values
            x_val = random.uniform(2.0, 5.0)
            y_val = random.uniform(2.0, 5.0)
            z_val = random.uniform(2.0, 5.0)
            
            x = ElementaryVar("x", x_val)
            y = ElementaryVar("y", y_val)
            z = ElementaryVar("z", z_val)
            
            # Create tree: 
            #       final
            #      /     \
            #   div1     div2
            #   /  \     /  \
            #  x    y   y    z
            
            div1 = Div(x, y)  # x/y
            div2 = Div(y, z)  # y/z
            final = Div(div1, div2)  # (x/y)/(y/z) = (x*z)/(y*y)
            
            # Compute: final = (x/y)/(y/z) = (x*z)/(y²)
            result = final.compute()
            expected_value = (x_val * z_val) / (y_val * y_val)
            self.assertAlmostEqual(result, expected_value)
            
            final.backward()
            
            # Calculate expected gradients using the chain rule:
            # For f = (x*z)/(y²)
            # df/dx = z/y² 
            # df/dy = -2*x*z/y³
            # df/dz = x/y²
            expected_x_grad = z_val / (y_val * y_val)
            expected_y_grad = -2 * x_val * z_val / (y_val**3)
            expected_z_grad = x_val / (y_val * y_val)
            
            self.assertAlmostEqual(x.grad, expected_x_grad)
            self.assertAlmostEqual(y.grad, expected_y_grad)
            self.assertAlmostEqual(z.grad, expected_z_grad)

    def test_gradient_involved(self):
        """
        Test gradient computation in a more complex scenario with multiple Div operations.
        Tests a mixture of variables and constants in division chains.
        """
        for _ in range(1000):
            # 1. Generate n from 3 to 10 
            n = random.randint(3, 10)
            
            # 2. Create n ElementaryVar objects with safe values (not too close to zero)
            variables = [ElementaryVar(f"var{i}", round(random.uniform(1.5, 5.0), 3)) for i in range(n + 1)]
            
            # Track numerator and denominator appearances for each variable
            counts = [0] * (n + 1)
            
            # 3. Initial setup with an intermediary value
            result = ConstantVar("start", 1)
            const_factor = 1.0  # Track constant factor
            
            # 4. Inner loop - divide by more variables/constants
            for _ in range(25):
                p = random.random()
                
                if p > 0.8:
                    # Choose a random variable to be in denominator (except the last one)
                    idx = random.randint(0, n - 1)
                    result = Div(result, variables[idx])
                    counts[idx] += 1
                else:
                    # Create a new constant to divide by (keep between 1.1-3.0 for stability)
                    const_val = random.uniform(1.1, 3.0)
                    const_var = ConstantVar("c", const_val)
                    result = Div(result, const_var)
                    const_factor *= const_val
                        
            # Compute the result
            result_value = result.compute()
            
            # make sure the result is computed correctly
            expected_result = 1.0
            
            var_factor = 1.0
            for i in range(n):
                if counts[i] > 0:
                    var_factor *= (variables[i].value ** counts[i])

            expected_result /= (var_factor * const_factor)
            self.assertAlmostEqual(result_value, expected_result)


            # Perform backward pass with gradient of 1.0
            result.backward()

            for i in range(n):
                if counts[i] > 0:
                    # the final result is 1 / (const_factor * var_i * counts[i] * (var_factor / (var_i ** counts[i]))) 
                    # computing (var_factor / (var_i ** counts[i])) as var_other_factor, we get 
                    # the gradient of the expression with respect to var_i is
                    # -(counts[i]) / (const_factor * var_factor * var_i)
                    expected_grad = -(counts[i]) / (const_factor * var_factor * variables[i].value) 
                    self.assertAlmostEqual(variables[i].grad, expected_grad)

                else:
                    # If variable wasn't used, its gradient should be None or 0
                    self.assertTrue(
                        variables[i].grad is None,
                        f"Variable {i} wasn't used but has gradient {variables[i].grad}"
                    )


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main()
