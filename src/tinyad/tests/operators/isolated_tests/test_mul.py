"""
This is a file to test the correctness of the multiplication operator
"""


import unittest
import random
import numpy as np 

from tinyad.autoDiff.operators.binary_ops import Mult
from tinyad.autoDiff.var import ConstantVar, ElementaryVar


class TestMultForwardBasic(unittest.TestCase):  
    """Test basic functionality of the Mult operator."""
    
    def test_init(self):
        """Test that Mult is initialized correctly."""
        x = ElementaryVar("x", 2.0)
        y = ElementaryVar("y", 3.0)
        mult = Mult(x, y)
        
        self.assertEqual(mult.name, "x*y")
        self.assertIsNone(mult.value)
        self.assertIsNone(mult.grad)
        self.assertEqual(mult.children, [x, y])
        self.assertEqual(mult.left, x)
        self.assertEqual(mult.right, y)
    
    
    def test_forward(self):
        """Test the forward method creates a new Mult instance with forwarded operands."""
        x = ElementaryVar("x", 2.0)
        y = ElementaryVar("y", 3.0)
        mult = Mult(x, y)
        
        # Since ElementaryVar.forward() returns self, the forwarded Mult should 
        # have the same operands but be a different instance
        forwarded = mult.forward()
        
        self.assertIsInstance(forwarded, Mult)
        self.assertIsNot(forwarded, mult)
        self.assertIs(forwarded.left, x)
        self.assertIs(forwarded.right, y)
    

    def test_call(self):
        """Test that calling a Mult instance invokes forward."""
        x = ElementaryVar("x", 2.0)
        y = ElementaryVar("y", 3.0)
        mult = Mult(x, y)
        
        # Called should be equivalent to forward
        called = mult()
        
        self.assertIsInstance(called, Mult)
        self.assertIsNot(called, mult)
        self.assertIs(called.left, x)
        self.assertIs(called.right, y)
    

    def test_compute(self):
        """Test the compute method with various random values."""
        # Test with 20 random pairs of values
        for _ in range(20):
            x_val = random.uniform(-100, 100)
            y_val = random.uniform(-100, 100)
            
            x = ElementaryVar("x", x_val)
            y = ElementaryVar("y", y_val)
            mult = Mult(x, y)
            
            # Compute should return x_val * y_val
            expected = x_val * y_val
            self.assertEqual(mult.compute(), expected)
            
            # Value should be cached after compute
            self.assertEqual(mult.value, expected)


    def test_nested_compute(self):
        """Test computation with nested Mult operators."""
        for _ in range(50):
            
            product_val = 1  # Start with 1 for multiplication
            result = Mult(ConstantVar("c", 1), ConstantVar("c", 1))

            for _ in range(10):
                val = random.uniform(-10, 10)  # Smaller range to avoid numeric overflow
                # make sure the mult works with the ConstantVar variable 
                result = Mult(result, ConstantVar("c", val))
                
                product_val *= val
                self.assertAlmostEqual(result.compute(), product_val)

                # make sure it works with ElementaryVar variables
                x = ElementaryVar("x", val)
                result = Mult(result, x)
                product_val *= val
                self.assertAlmostEqual(result.compute(), product_val)

    def test_mult_with_constants(self):
        """Test multiplying variables with constants."""
        for _ in range(10):
            x_val = random.uniform(-100, 100)
            const_val = random.uniform(-100, 100)
            
            x = ElementaryVar("x", x_val)
            const = ConstantVar("c", const_val)
            mult = Mult(x, const)
            
            # Test computation
            self.assertEqual(mult.compute(), x_val * const_val)
            
            # Test gradient - should flow to the ElementaryVar scaled by the constant
            mult.backward()  # Default gradient of 1.0
            self.assertEqual(x.grad, const_val)  # x's gradient is y's value
            self.assertEqual(const.grad, 0)  # Constants always have zero gradient

    def test_multi_with_different_values(self):
        """Test multiplication with different values."""
        for _ in range(10):
            prod = 1
            result = ConstantVar("c", 1)

            for _ in range(10):
                val = random.uniform(-10, 10)
                result = Mult(result, ConstantVar("c", val))
                prod *= val
                self.assertAlmostEqual(result.compute(), prod)

            



class TestMultBackwardBasic(unittest.TestCase):
    """Test gradient computation for the Mult operator."""
    
    def test_simple_gradient(self):
        """Test gradient computation with two ElementaryVars."""
        for i in range(1, 11):
            for j in range(1, 11):
                x = ElementaryVar("x", i)
                y = ElementaryVar("y", j)
                mult = Mult(x, y)
                
                # Forward and backward pass
                mult.compute()  # Ensure value is computed
                mult.backward()  # Default gradient of 1.0
                
                # For z = x * y:
                # dz/dx = y, dz/dy = x
                self.assertEqual(mult.grad, 1.0)
                self.assertEqual(x.grad, j)  # x's gradient = y's value
                self.assertEqual(y.grad, i)  # y's gradient = x's value
                
                # Test with a different upstream gradient
                x.grad = None
                y.grad = None
                mult.grad = None
                
                value = round(10 * random.random(), 3)
                mult.backward(value)
                
                self.assertEqual(mult.grad, value)
                self.assertEqual(x.grad, value * j)  # Scaled by upstream gradient
                self.assertEqual(y.grad, value * i)  # Scaled by upstream gradient


    def test_different_upstream_gradients(self):
        """Test backpropagation with different upstream gradients."""
        for _ in range(10):
            x_val = random.uniform(0.1, 10)  # Avoid zeros
            y_val = random.uniform(0.1, 10)  # Avoid zeros
            
            x = ElementaryVar("x", x_val)
            y = ElementaryVar("y", y_val)
            
            mult = Mult(x, y)
            
            # Use different upstream gradients
            gradient1 = random.uniform(0.1, 10)
            gradient2 = random.uniform(0.1, 10)
            
            # Apply first gradient
            mult.compute()
            mult.backward(gradient1)
            
            x_grad1 = x.grad
            y_grad1 = y.grad
            
            # Reset and apply second gradient
            x.grad = None
            y.grad = None
            mult.grad = None
            
            mult.backward(gradient2)
            
            # Check proportionality
            self.assertAlmostEqual(x_grad1 / x.grad, gradient1 / gradient2)
            self.assertAlmostEqual(y_grad1 / y.grad, gradient1 / gradient2)

    
    def test_gradient_accumulation(self):
        """Test that gradients accumulate properly in shared variables."""
        x = ElementaryVar("x", 2.0)
        y = ElementaryVar("y", 3.0)
        z = ElementaryVar("z", 4.0)
        
        # Create two Mult operations that share variable x
        mult1 = Mult(x, y)
        mult2 = Mult(x, z)
        
        # Compute both and backpropagate
        mult1.compute()
        mult2.compute()
        
        mult1.backward()
        mult2.backward()
        
        # x should receive gradient from both operations
        self.assertEqual(x.grad, y.compute() + z.compute())  # 3 + 4 = 12
        self.assertEqual(y.grad, x.compute())  # 2
        self.assertEqual(z.grad, x.compute())  # 2


    def test_zero_values(self):
        """Test multiplication with zero values."""
        # x = 0, y = 5
        x = ElementaryVar("x", 0.0)
        y = ElementaryVar("y", 5.0)
        mult = Mult(x, y)
        
        # Result should be 0
        self.assertEqual(mult.compute(), 0.0)
        
        # Backprop
        mult.backward()
        
        # y's gradient is x's value (0)
        self.assertEqual(y.grad, 0.0)
        # x's gradient is y's value (5)
        self.assertEqual(x.grad, 5.0)
        
        # x = 3, y = 0
        x = ElementaryVar("x", 3.0)
        y = ElementaryVar("y", 0.0)
        mult = Mult(x, y)
        
        # Result should be 0
        self.assertEqual(mult.compute(), 0.0)
        
        # Backprop
        mult.backward()
        
        # y's gradient is x's value (3)
        self.assertEqual(y.grad, 3.0)
        # x's gradient is y's value (0)
        self.assertEqual(x.grad, 0.0)


    def test_numerical_stability(self):
        """Test Mult with very large and very small numbers."""
        # Test with very large numbers
        x_large = ElementaryVar("x_large", 1e10)
        y_large = ElementaryVar("y_large", 1e10)
        mult_large = Mult(x_large, y_large)
        self.assertAlmostEqual(mult_large.compute(), 1e20)
        
        # Test with very small numbers
        x_small = ElementaryVar("x_small", 1e-10)
        y_small = ElementaryVar("y_small", 1e-10)
        mult_small = Mult(x_small, y_small)
        self.assertAlmostEqual(mult_small.compute(), 1e-20)
        
        # Test with mix of large and small
        mult_mixed = Mult(x_large, x_small)
        self.assertAlmostEqual(mult_mixed.compute(), 1)


    def test_complex_expression_tree(self):
        """Test gradients in a more complex expression tree with branches."""
        for _ in range(1000):
            x = ElementaryVar("x", random.uniform(0.5, 2))
            y = ElementaryVar("y", random.uniform(0.5, 2))
            z = ElementaryVar("z", random.uniform(0.5, 2))
            
            # Create tree: 
            #       final
            #      /     \
            #   mult1    mult2
            #   /  \     /  \
            #  x    y   y    z
            
            mult1 = Mult(x, y)
            mult2 = Mult(y, z)
            final = Mult(mult1, mult2)
            
            # Compute: final = (x*y) * (y*z) = x*y²*z
            final.compute()
            expected_value = x.value * y.value ** 2 * z.value

            self.assertAlmostEqual(final.compute(), expected_value)
            
            final.backward()
            
            expected_x_grad = y.value ** 2 * z.value
            expected_y_grad = x.value * 2 * y.value * z.value
            expected_z_grad = x.value * y.value ** 2

            self.assertAlmostEqual(x.grad, expected_x_grad)
            self.assertAlmostEqual(y.grad, expected_y_grad)
            self.assertAlmostEqual(z.grad, expected_z_grad)


    def test_gradient_involved(self):
        """
        Test gradient computation in a more complex scenario with multiple Mult operations.
        Tests a mixture of variables and constants in multiplication chains.
        """
        for _ in range(1000):
            # 1. Generate n from 3 to 10
            n = random.randint(3, 10)
            
            # 2. Create n ElementaryVar objects
            variables = [ElementaryVar(f"var{i}", round(random.uniform(0.5, 2), 3)) for i in range(n + 1)]
            # Track how many times each variable is used in Mult operations
            counts = [0] * (n + 1)
            
            # 3. Initial setup: start with constant 1
            const_product = 1.0
            result = ConstantVar("c", 1.0)
            
            # 4. Inner loop - multiply more variables/constants to the expression
            for _ in range(20):
                # 5. Choose between variable or constant
                p = random.random()
                
                if p > 0.75:
                    # Choose a random variable to multiply (except the last one) to guarantee that at least one variable isn't used
                    idx = random.randint(0, n - 1)
                    result = Mult(result, variables[idx])
                    counts[idx] += 1
                else:
                    # Create a new constant to multiply
                    const_val = random.uniform(0.5, 2.0)  # Avoid extreme values, and round to 3 decimal places
                    const_var = ConstantVar("c", const_val)
                    result = Mult(result, const_var)
                    const_product *= const_val
            
            # Compute the result
            result_value = result.compute()
            
            # Perform backward pass with gradient of 1.0
            result.backward()
            
            # Calculate the product of all variables (with appropriate powers)
            var_product = 1.0
            for i in range(n):
                var_product *= variables[i].value ** counts[i]
            
            # The result should be const_product * var_product
            self.assertAlmostEqual(result_value, const_product * var_product)
            
            # Check gradients: For each variable, gradient should be:
            # const_product * (product of all other variable terms) * count
            for i in range(n + 1):
                if counts[i] > 0:
                    # Calculate expected gradient
                    expected_gradient = const_product * (var_product / (variables[i].value)) * counts[i]
                    
                    # Compare with actual gradient
                    self.assertAlmostEqual(
                        variables[i].grad, 
                        expected_gradient,
                        msg=f"Variable {i} gradient incorrect. Expected {expected_gradient}, got {variables[i].grad}"
                    )
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
