"""
This is a file to test the correctness of the add operator
"""


import unittest
import random

from tinyad.autoDiff.operators.binary_ops import Add
from tinyad.autoDiff.var import ElementaryVar


class TestAddBasic(unittest.TestCase):
    """Test basic functionality of the Add operator."""
    
    def test_init(self):
        """Test that Add is initialized correctly."""
        x = ElementaryVar("x", 2.0)
        y = ElementaryVar("y", 3.0)
        add = Add(x, y)
        
        self.assertEqual(add.name, "+")
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
        for _ in range(10):
            a_val = random.uniform(-100, 100)
            b_val = random.uniform(-100, 100)
            c_val = random.uniform(-100, 100)
            
            a = ElementaryVar("a", a_val)
            b = ElementaryVar("b", b_val)
            c = ElementaryVar("c", c_val)
            
            # Create (a + b) + c
            add1 = Add(a, b)
            add2 = Add(add1, c)
            
            expected = a_val + b_val + c_val
            self.assertEqual(add2.compute(), expected)


class TestAddGrad(unittest.TestCase):
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
                self.assertEqual(variables[i].grad, counts[i], 
                                f"Variable {i} gradient {variables[i].grad} does not match count {counts[i]}")

if __name__ == '__main__':
    unittest.main()
