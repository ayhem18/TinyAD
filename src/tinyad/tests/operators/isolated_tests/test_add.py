"""
This is a file to test the correctness of the add operator
"""


import unittest
import random

from tinyad.autoDiff.operators.binary_ops import Add
from tinyad.autoDiff.var import ConstantVar, ElementaryVar


class TestAddForwardBasic(unittest.TestCase):  
    """Test basic functionality of the Add operator."""
    
    def test_init(self):
        """Test that Add is initialized correctly."""
        x = ElementaryVar("x", 2.0)
        y = ElementaryVar("y", 3.0)
        add = Add(x, y)
        
        self.assertEqual(add.name, "x+y")
        self.assertIsNone(add.value)
        self.assertIsNone(add.grad)
        self.assertEqual(add.children, [x, y])
        self.assertEqual(add.left, x)
        self.assertEqual(add.right, y)
    

    def test_compute(self):
        """Test the compute method with various random values."""
        # Test with 20 random pairs of values
        for _ in range(20):
            x_val = random.uniform(-100, 100)
            y_val = random.uniform(-100, 100)
            
            x = ElementaryVar("x", x_val)
            y = ElementaryVar("y", y_val)
            add = Add(x, y)
            
            # Compute should return x_val + y_val
            expected = x_val + y_val
            self.assertEqual(add.compute(), expected)
            
            # Value should be cached after compute
            self.assertEqual(add.value, expected)
    
    
    def test_forward(self):
        """Test the forward method creates a new Add instance with forwarded operands."""
        x = ElementaryVar("x", 2.0)
        y = ElementaryVar("y", 3.0)
        add = Add(x, y)
        
        # Since ElementaryVar.forward() returns self, the forwarded Add should 
        # have the same operands but be a different instance
        forwarded = add.forward()
        
        self.assertIsInstance(forwarded, Add)
        self.assertIsNot(forwarded, add)
        self.assertIs(forwarded.left, x)
        self.assertIs(forwarded.right, y)
    
    
    def test_call(self):
        """Test that calling an Add instance invokes forward."""
        x = ElementaryVar("x", 2.0)
        y = ElementaryVar("y", 3.0)
        add = Add(x, y)
        
        # Called should be equivalent to forward
        called = add()
        
        self.assertIsInstance(called, Add)
        self.assertIsNot(called, add)
        self.assertIs(called.left, x)
        self.assertIs(called.right, y)
    

    def test_nested_compute(self):
        """Test computation with nested Add operators."""
        for _ in range(50):
            
            s = 0
            result = Add(ConstantVar("c", 0), ConstantVar("c", 0))

            for _ in range(10):
                val = random.uniform(-100, 100)
                # make sure the add works with the ConstantVar variable 
                result = Add(result, ConstantVar("c", val))
                
                s += val
                self.assertEqual(result.compute(), s)

                # make sure it works with ElementaryVar variables
                x = ElementaryVar("x", val)
                result = Add(result, x)
                s += val
                self.assertEqual(result.compute(), s)



    def test_add_with_constants(self):
        """Test adding variables with constants."""
        for _ in range(10):
            x_val = random.uniform(-100, 100)
            const_val = random.uniform(-100, 100)
            
            x = ElementaryVar("x", x_val)
            const = ConstantVar("c", const_val)
            add = Add(x, const)
            
            # Test computation
            self.assertEqual(add.compute(), x_val + const_val)
            
            # Test gradient - should only flow to the ElementaryVar, not the constant
            add.backward()
            self.assertEqual(x.grad, 1.0)
            self.assertEqual(const.grad, 0)


class TestAddBackwardBasic(unittest.TestCase):
    """Test gradient computation for the Add operator."""
    
    def test_simple_gradient(self):
        """Test gradient computation with two ElementaryVars."""
        for i in range(1, 11):
            for j in range(1, 11):
                x = ElementaryVar("x", i)
                y = ElementaryVar("y", j)
                add = Add(x, y)
                
                # Forward and backward pass
                add.compute()  # Ensure value is computed
                add.backward()  # Default gradient of 1.0
                
                # For z = x + y:
                # dz/dx = 1, dz/dy = 1 (regardless of the values of x and y)
                self.assertEqual(add.grad, 1.0)
                self.assertEqual(x.grad, 1.0)
                self.assertEqual(y.grad, 1.0)
                
                # Test with a different upstream gradient
                x.grad = None
                y.grad = None
                add.grad = None
                
                value = random.random()
                add.backward(value)
                
                self.assertEqual(add.grad, value)
                self.assertEqual(x.grad, value)
                self.assertEqual(y.grad, value)


    def test_different_upstream_gradients(self):
        """Test backpropagation with different upstream gradients."""
        for _ in range(10):
            x = ElementaryVar("x", random.uniform(-10, 10))
            y = ElementaryVar("y", random.uniform(-10, 10))
            
            # Create a chain: x + y -> result1 -> result2
            result1 = Add(x, y)
            
            # Use different upstream gradients
            gradient1 = random.uniform(0.1, 10)
            gradient2 = random.uniform(0.1, 10)
            
            # Apply first gradient
            result1.compute()
            result1.backward(gradient1)
            
            x_grad1 = x.grad
            y_grad1 = y.grad
            
            # Reset and apply second gradient
            x.grad = None
            y.grad = None
            result1.grad = None
            
            result1.backward(gradient2)
            
            # Check proportionality
            self.assertAlmostEqual(x_grad1 / x.grad, gradient1 / gradient2)
            self.assertAlmostEqual(y_grad1 / y.grad, gradient1 / gradient2)


    def test_numerical_stability(self):
        """Test Add with very large and very small numbers."""
        # Test with very large numbers
        x_large = ElementaryVar("x_large", 1e15)
        y_large = ElementaryVar("y_large", 1e15)
        add_large = Add(x_large, y_large)
        self.assertEqual(add_large.compute(), 2e15)
        
        # Test with very small numbers
        x_small = ElementaryVar("x_small", 1e-15)
        y_small = ElementaryVar("y_small", 1e-15)
        add_small = Add(x_small, y_small)
        self.assertEqual(add_small.compute(), 2e-15)
        
        # Test with mix of large and small
        add_mixed = Add(x_large, x_small)
        self.assertEqual(add_mixed.compute(), 1e15 + 1e-15)


    def test_complex_expression_tree(self):
        """Test gradients in a more complex expression tree with branches."""
        for _ in range(100):
            x = ElementaryVar("x", random.uniform(-10, 10))
            y = ElementaryVar("y", random.uniform(-10, 10))
            z = ElementaryVar("z", random.uniform(-10, 10))
            
            # Create tree: 
            #       final
            #      /     \
            #   add1     add2
            #   /  \     /  \
            #  x    y   y    z
            
            add1 = Add(x, y)
            add2 = Add(y, z)
            final = Add(add1, add2)
            
            final.compute()
            final.backward()
            
            self.assertEqual(x.grad, 1.0)
            self.assertEqual(y.grad, 2.0)
            self.assertEqual(z.grad, 1.0)


    def test_gradient_involved(self):
        """
        Test gradient computation in a more complex scenario with multiple Add operations.
        Verifies that gradient flows correctly through a chain of additions.
        """
        for _ in range(10):
            # 1. Generate n from 3 to 10
            n = random.randint(3, 10)
            
            # 2. Create n ElementaryVar objects
            variables = [ElementaryVar(f"var{i}", random.uniform(-10, 10)) for i in range(n)]
            
            # Track how many times each variable is used in Add operations
            counts = [0] * n
            
            # 3. Create an initial Add using the first two variables
            result = Add(variables[0], variables[1])
            counts[0] += 1
            counts[1] += 1
            
            # 4. Inner loop - add more variables to the expression
            for _ in range(50):
                # 5. Choose a random variable to add
                idx = random.randint(0, n - 1)
                
                # Update the expression: result = result + variables[idx]
                result = Add(result, variables[idx])
                counts[idx] += 1
            
            # Compute the result
            result.compute()
            
            # Perform backward pass with gradient of 1.0
            result.backward()
            
            # 6. Check that each variable's gradient matches its usage count
            for i in range(n):
                if counts[i] > 0:   
                    self.assertEqual(variables[i].grad, counts[i], 
                                    f"Variable {i} gradient {variables[i].grad} does not match count {counts[i]}")
                else:
                    self.assertIsNone(variables[i].grad)



if __name__ == '__main__':
    unittest.main()
