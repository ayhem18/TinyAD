import random
import unittest
import numpy as np

from tinyad.autoDiff.operators.unary_ops import AbsVal, Neg
from tinyad.autoDiff.var import ElementaryVar


class TestNegAbsCombinations(unittest.TestCase):

    def test_negation_absolute_identities(self):
        """Test mathematical identities involving negation and absolute value."""
        for _ in range(1000):
            x_val = random.uniform(-100, 100)
            x = ElementaryVar("x", x_val)
            
            # Identity 1: |(-x)| = |x|
            identity1_left = AbsVal(Neg(x))
            identity1_right = AbsVal(x)
            
            self.assertEqual(identity1_left.compute(), identity1_right.compute())
            
            # Identity 2: -|x| is just negation after absolute value
            identity2 = Neg(AbsVal(x))
            self.assertEqual(identity2.compute(), -abs(x_val))
            
            # Identity 3: |-(|x|)| = |x| (absolute value of the negation of absolute value)
            identity3 = AbsVal(Neg(AbsVal(x)))
            self.assertEqual(identity3.compute(), abs(x_val))

    def test_negation_absolute_gradients(self):
        """Test gradient computation for combined negation and absolute value operations."""
        for _ in range(1000):
            x_val = random.uniform(-100, 100)
            x = ElementaryVar("x", x_val)
            
            # Case 1: |(-x)| - gradient should be sign(x) (since |-x| is the same as |x|)
            expr1 = AbsVal(Neg(x))
            expr1.compute()
            expr1.backward()
            expected_grad1 = np.sign(x_val)
            self.assertEqual(x.grad, expected_grad1)
            
            # Reset gradient
            x.grad = None
            
            # Case 2: -|x| - gradient should be -sign(x)
            expr2 = Neg(AbsVal(x))
            expr2.compute()
            expr2.backward()
            expected_grad2 = -np.sign(x_val)
            self.assertEqual(x.grad, expected_grad2)

    def test_multiple_absolute_negation_composition(self):
        """Test multiple compositions of negation and absolute value."""
        for _ in range(5000):
            x_val = round(random.uniform(-100, 100), 4)
            x = ElementaryVar("x", x_val)
            
            # Create a chain of alternating Neg and AbsVal operations
            n_ops = random.randint(3, 50)
            ops = [random.choice([AbsVal, Neg]) for _ in range(n_ops)]
            
            # Apply operations in sequence
            result = x
            for op in ops:
                result = op(result)
            
            # Compute the result
            computed = result.compute()
            
            # Calculate expected result by simulating operations
            expected = x_val

            for op in ops:
                if op == AbsVal:
                    expected = abs(expected)
                elif op == Neg:
                    expected = -expected
            self.assertEqual(computed, expected, "the forward pass is buggy")

            # the resulting function can be computed in a closed form
            # find the last index of an AbsVal operation
            last_abs_val_idx = -1
            for i in range(len(ops) - 1, -1, -1):
                if ops[i] == AbsVal:
                    last_abs_val_idx = i
                    break
                
            num_negations = n_ops - 1 - last_abs_val_idx
            coeff = 1 if num_negations % 2 == 0 else -1
            
            
            if last_abs_val_idx == -1: 
                # this is an extreme case where no AbsVal operation is present 
                # the resulting function is just a polynomial
                res = x_val * coeff
                self.assertEqual(expected, res, f"the alternative computation for the extreme case is buggy. Found: {res}, expected: {expected}") 
                result.backward()
                expected_grad = coeff
                self.assertEqual(x.grad, expected_grad, f"the gradient for the extreme case is buggy. Found: {x.grad}, expected: {expected_grad}")

                # make sure to move to the next iteration.
                continue
    
            # at this point, we know that there is at least one AbsVal operation 
            res = abs(x_val)  * coeff
            self.assertEqual(expected, res, f"the alternative computation is buggy. Found: {res}, expected: {expected}") 
            # Test gradient
            result.backward()
            expected_grad = np.sign(x_val).item() * coeff
            self.assertEqual(x.grad, expected_grad, f"the gradient is buggy. Found: {x.grad}, expected: {expected_grad}")


    def test_zero_crossing_behavior(self):
        """Test behavior of negation and absolute value around x=0."""
        # Create a small positive value
        epsilon = 1e-6
        
        # Test positive epsilon
        x_pos = ElementaryVar("x_pos", epsilon)
        expr_pos = Neg(AbsVal(x_pos))
        expr_pos.compute()
        expr_pos.backward()
        self.assertEqual(x_pos.grad, -1)  # Since x > 0, sign(x) = 1
        
        # Test negative epsilon
        x_neg = ElementaryVar("x_neg", -epsilon)
        expr_neg = Neg(AbsVal(x_neg))
        expr_neg.compute()
        expr_neg.backward()
        self.assertEqual(x_neg.grad, 1)  # Since x < 0, sign(x) = -1
        
        # Test exactly zero
        x_zero = ElementaryVar("x_zero", 0.0)
        expr_zero = Neg(AbsVal(x_zero))
        expr_zero.compute()
        expr_zero.backward()
        self.assertEqual(x_zero.grad, 0)  # By our implementation choice


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main()
