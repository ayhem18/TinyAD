import unittest

from tinyad.autoDiff.var import ElementaryVar, ConstantVar
from tinyad.autoDiff.common import NUM


class TestElementaryVar(unittest.TestCase):
    """Test class for the ElementaryVar implementation."""
    
    def test_init(self):
        """Test that the ElementaryVar is initialized correctly."""
        var = ElementaryVar("x", 5.0)
        self.assertEqual(var.name, "x")
        self.assertEqual(var.value, 5.0)
        self.assertIsNone(var.grad)
        self.assertEqual(var.children, [])
    
    def test_forward(self):
        """Test that forward returns the var itself."""
        var = ElementaryVar("x", 5.0)
        forward_var = var.forward()
        self.assertIs(forward_var, var)
    
    def test_call(self):
        """Test that calling the var invokes forward."""
        var = ElementaryVar("x", 5.0)
        called_var = var()
        self.assertIs(called_var, var)
    
    def test_backward_initial(self):
        """Test backward with initial gradient."""
        var = ElementaryVar("x", 5.0)
        var.backward(2.0)
        self.assertEqual(var.grad, 2.0)
    
    def test_backward_accumulation(self):
        """Test that backward accumulates gradients."""
        var = ElementaryVar("x", 5.0)
        var.backward(2.0)
        var.backward(3.0)
        self.assertEqual(var.grad, 5.0)
    
    def test_backward_default(self):
        """Test backward with default value."""
        var = ElementaryVar("x", 5.0)
        var.backward()
        self.assertEqual(var.grad, 1)
    
    def test_compute(self):
        """Test that compute returns the stored value."""
        var = ElementaryVar("x", 5.0)
        self.assertEqual(var.compute(), 5.0)


class TestConstantVar(unittest.TestCase):
    """Test class for the ConstantVar implementation."""
    
    def test_init(self):
        """Test that the ConstantVar is initialized correctly."""
        const = ConstantVar("c", 3.0)
        self.assertEqual(const.name, "c")
        self.assertEqual(const.value, 3.0)
        self.assertIsNone(const.grad)
        self.assertEqual(const.children, [])
    
    def test_forward(self):
        """Test that forward returns the var itself."""
        const = ConstantVar("c", 3.0)
        forward_const = const.forward()
        self.assertIs(forward_const, const)
    
    def test_call(self):
        """Test that calling the var invokes forward."""
        const = ConstantVar("c", 3.0)
        called_const = const()
        self.assertIs(called_const, const)
    
    def test_backward(self):
        """Test backward always sets gradient to 0."""
        const = ConstantVar("c", 3.0)
        # Test with a value
        const.backward(2.0)
        self.assertEqual(const.grad, 0)
        
        # Test with default value
        const = ConstantVar("c", 3.0)
        const.backward()
        self.assertEqual(const.grad, 0)
    
    def test_compute(self):
        """Test that compute returns the stored value."""
        const = ConstantVar("c", 3.0)
        self.assertEqual(const.compute(), 3.0)


if __name__ == '__main__':
    unittest.main()
