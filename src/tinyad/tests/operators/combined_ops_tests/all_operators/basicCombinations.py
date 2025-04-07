import random
import unittest

import numpy as np

from tinyad.autoDiff.operators.binary_ops import Add, Mult, Sub, Div, Exp
from tinyad.autoDiff.operators.unary_ops import AbsVal, Neg
from tinyad.autoDiff.var import ElementaryVar, ConstantVar
from tinyad.tests.operators.combined_ops_tests.binary_operators.bin_ops_base_test import BinaryOperatorsBaseTest


class TestUnaryBinaryBasicCombinations(BinaryOperatorsBaseTest):
    """Test class for combined operations of Unary and Binary operators.""" 


    ########################### test abs value with other operators ###########################
    def test_absval_mult(self):
        """
        Test that |x₁ * x₂ * ... * xₙ| behaves exactly like |x₁| * |x₂| * ... * |xₙ|.
        This is a mathematical identity derived from |x * y| = |x| * |y|.
        """
        for _ in range(1000):
            # Generate a random number of variables (2-6)
            n_vars = random.randint(2, 10)
            
            # Create variables with mixed positive and negative values
            variables = [ElementaryVar(f"x{i}", random.uniform(-10, 10)) for i in range(n_vars)]
            
            for i in range(n_vars):
                if variables[i].value == 0:
                    variables[i] = ElementaryVar(f"x{i}", random.uniform(0.1, 10))

            # Create a copy for the second expression to have independent gradients
            variables2 = [ElementaryVar(f"y{i}", variables[i].value) for i in range(n_vars)]
            
            for i in range(n_vars):
                if variables2[i].value == 0:
                    variables2[i] = ElementaryVar(f"y{i}", random.uniform(0.1, 10))

            # Build the product expression: x₁ * x₂ * ... * xₙ
            product = variables[0]
            for i in range(1, n_vars):
                product = Mult(product, variables[i])
            
            # Expression 1: |product|
            abs_product = AbsVal(product)
            
            # Expression 2: |x₁| * |x₂| * ... * |xₙ|
            product_of_abs = AbsVal(variables2[0])
            for i in range(1, n_vars):
                product_of_abs = Mult(product_of_abs, AbsVal(variables2[i]))
            
            # Compute and verify both expressions have same value
            abs_product_value = abs_product.compute()
            product_of_abs_value = product_of_abs.compute()
            
            self.assertAlmostEqual(abs_product_value, product_of_abs_value)
            
            # Perform backward pass on both expressions
            abs_product.backward()
            product_of_abs.backward()
            
            # calculate the gradients of the variables
            # since the expression is |x_i| * |other_vars_product|
            # the gradient of the expression with respect to x_i is |other_vars_product| * sign(x_i)
            # and the gradient of the expression with respect to other_vars_product is x_i * sign(x_i)
            
            # the code above guarantees that x_i is not 0, so we can compute the product all but x_i efficiently
            product_but_xi = [product_of_abs.compute() / abs(variables[i].value) for i in range(n_vars)]            
        
            for i in range(n_vars):
                self.assertAlmostEqual(variables[i].grad, product_but_xi[i] * np.sign(variables[i].value).item())
                self.assertAlmostEqual(variables2[i].grad, product_but_xi[i] * np.sign(variables[i].value).item())


    # def test_absval_div(self):
    #     """
    #     Test that |x₁*x₂*...*xₙ/y₁*y₂*...*yₙ| behaves exactly like |x₁*x₂*...*xₙ|/|y₁*y₂*...*yₙ|.
    #     This is a mathematical identity: |a/b| = |a|/|b|
    #     """
    #     for _ in range(1000):
    #         # Generate random numbers of numerator and denominator variables
    #         num_vars = random.randint(1, 4)
    #         denom_vars = random.randint(1, 4)
            
    #         # Create variables with mixed signs, avoiding values close to zero for denominator
    #         numerator_vars = [ElementaryVar(f"x{i}", random.uniform(-10, 10)) for i in range(num_vars)]
    #         denominator_vars = [ElementaryVar(f"y{i}", random.uniform(-10, -0.5) if random.random() < 0.5 else random.uniform(0.5, 10)) 
    #                            for i in range(denom_vars)]
            
    #         # Create copies for the second expression
    #         numerator_vars2 = [ElementaryVar(f"x2_{i}", numerator_vars[i].value) for i in range(num_vars)]
    #         denominator_vars2 = [ElementaryVar(f"y2_{i}", denominator_vars[i].value) for i in range(denom_vars)]
            
    #         # Build the numerator and denominator expressions
    #         numerator = numerator_vars[0] if num_vars > 0 else ConstantVar("one", 1.0)
    #         for i in range(1, num_vars):
    #             numerator = Mult(numerator, numerator_vars[i])
                
    #         denominator = denominator_vars[0] if denom_vars > 0 else ConstantVar("one", 1.0)
    #         for i in range(1, denom_vars):
    #             denominator = Mult(denominator, denominator_vars[i])
            
    #         # Expression 1: |numerator/denominator|
    #         ratio = Div(numerator, denominator)
    #         abs_ratio = AbsVal(ratio)
            
    #         # Expression 2: |numerator|/|denominator|
    #         numerator2 = numerator_vars2[0] if num_vars > 0 else ConstantVar("one", 1.0)
    #         for i in range(1, num_vars):
    #             numerator2 = Mult(numerator2, numerator_vars2[i])
                
    #         denominator2 = denominator_vars2[0] if denom_vars > 0 else ConstantVar("one", 1.0)
    #         for i in range(1, denom_vars):
    #             denominator2 = Mult(denominator2, denominator_vars2[i])
            
    #         abs_numerator = AbsVal(numerator2)
    #         abs_denominator = AbsVal(denominator2)
    #         ratio_of_abs = Div(abs_numerator, abs_denominator)
            
    #         # Compute and verify both expressions have same value
    #         abs_ratio_value = abs_ratio.compute()
    #         ratio_of_abs_value = ratio_of_abs.compute()
            
    #         self.assertAlmostEqual(abs_ratio_value, ratio_of_abs_value)
            
    #         # Perform backward pass on both expressions
    #         abs_ratio.backward()
    #         ratio_of_abs.backward()
            
    #         # Verify gradients have correct magnitudes
    #         # This is complex due to chain rule, but values should be close for identically valued variables
    #         for i in range(num_vars):
    #             # The magnitudes should be the same but signs may differ
    #             self.assertAlmostEqual(abs(numerator_vars[i].grad), abs(numerator_vars2[i].grad))
            
    #         for i in range(denom_vars):
    #             self.assertAlmostEqual(abs(denominator_vars[i].grad), abs(denominator_vars2[i].grad))

    # def test_absval_add_2_positives(self):
    #     """
    #     Test that |x + y| behaves the same as x + y when x and y are guaranteed to be positive.
    #     When both operands are positive, the absolute value doesn't change anything.
    #     """
    #     for _ in range(1000):
    #         # Generate positive random values
    #         x_val = random.uniform(0.1, 10)
    #         y_val = random.uniform(0.1, 10)
            
    #         x1 = ElementaryVar("x1", x_val)
    #         y1 = ElementaryVar("y1", y_val)
            
    #         x2 = ElementaryVar("x2", x_val)
    #         y2 = ElementaryVar("y2", y_val)
            
    #         # Expression 1: |x + y|
    #         sum_expr = Add(x1, y1)
    #         abs_sum = AbsVal(sum_expr)
            
    #         # Expression 2: x + y (no absolute value needed since both are positive)
    #         simple_sum = Add(x2, y2)
            
    #         # Compute and verify
    #         abs_sum_value = abs_sum.compute()
    #         simple_sum_value = simple_sum.compute()
            
    #         self.assertAlmostEqual(abs_sum_value, simple_sum_value)
            
    #         # Backward pass
    #         abs_sum.backward()
    #         simple_sum.backward()
            
    #         # For positive values, gradient of |x + y| should be the same as gradient of x + y
    #         self.assertEqual(x1.grad, x2.grad)
    #         self.assertEqual(y1.grad, y2.grad)
            
    #         # Also verify that both gradients are 1 (since d/dx(x+y) = d/dy(x+y) = 1)
    #         self.assertEqual(x1.grad, 1.0)
    #         self.assertEqual(y1.grad, 1.0)

    # def test_absval_add_pos_neg_sum_pos(self):
    #     """
    #     Test that |x + y| behaves the same as x + y when x + y is positive (testing the case where x and y have opposite signs).
    #     """
    #     for _ in range(1000):
    #         x_val = random.uniform(20, 50)
    #         y_val = random.uniform(-20, -1)
            
    #         x1 = ElementaryVar("x1", x_val)
    #         y1 = ElementaryVar("y1", y_val)
            
    #         x2 = ElementaryVar("x2", x_val)
    #         y2 = ElementaryVar("y2", y_val)
            
    #         # Expression 1: |x + y|
    #         sum_expr = Add(x1, y1)
    #         abs_sum = AbsVal(sum_expr)

    #         # Expression 2: x + y
    #         simple_sum = Add(x2, y2)
            
    #         # Compute and verify
    #         abs_sum_value = abs_sum.compute()
    #         simple_sum_value = simple_sum.compute()
            
    #         self.assertAlmostEqual(abs_sum_value, simple_sum_value)
            
    #         # Backward pass
    #         abs_sum.backward()
        
    # def test_absval_sub(self):
    #     """
    #     Test that |x - y| behaves the same as x - y when x > y and behaves like y - x when x < y.
    #     This tests the piecewise behavior of absolute value.
    #     """
    #     for _ in range(1000):
    #         # Generate two random values where one is larger than the other
    #         val1 = random.uniform(0.1, 10)
    #         val2 = random.uniform(0.1, 10)
            
    #         # Ensure val1 and val2 are different enough to avoid numerical issues
    #         while abs(val1 - val2) < 1e-6:
    #             val1 = random.uniform(0.1, 10)
    #             val2 = random.uniform(0.1, 10)
            
    #         # Determine which value is larger
    #         x_larger_than_y = val1 > val2
    #         x_val = max(val1, val2)
    #         y_val = min(val1, val2)
            
    #         # Create variables
    #         x1 = ElementaryVar("x1", x_val)
    #         y1 = ElementaryVar("y1", y_val)
            
    #         x2 = ElementaryVar("x2", x_val)
    #         y2 = ElementaryVar("y2", y_val)
            
    #         # Expression 1: |x - y|
    #         diff = Sub(x1, y1)
    #         abs_diff = AbsVal(diff)
            
    #         # Expression 2: depends on which value is larger
    #         expected_diff = Sub(x2, y2)  # When x > y, |x - y| = x - y
            
    #         # Compute and verify
    #         abs_diff_value = abs_diff.compute()
    #         expected_diff_value = expected_diff.compute()
            
    #         self.assertAlmostEqual(abs_diff_value, expected_diff_value)
            
    #         # Backward pass
    #         abs_diff.backward()
    #         expected_diff.backward()
            
    #         # Verify gradients
    #         # When x > y, the gradients are:
    #         # d/dx|x-y| = d/dx(x-y) = 1
    #         # d/dy|x-y| = d/dy(x-y) = -1
    #         self.assertEqual(x1.grad, 1.0)
    #         self.assertEqual(y1.grad, -1.0)
    #         self.assertEqual(x2.grad, 1.0)
    #         self.assertEqual(y2.grad, -1.0)
            
    #         # Now test the opposite case: x < y
    #         x_val = min(val1, val2)
    #         y_val = max(val1, val2)
            
    #         # Reset all variables
    #         x1 = ElementaryVar("x1", x_val)
    #         y1 = ElementaryVar("y1", y_val)
            
    #         x2 = ElementaryVar("x2", x_val)
    #         y2 = ElementaryVar("y2", y_val)
            
    #         # Expression 1: |x - y|
    #         diff = Sub(x1, y1)
    #         abs_diff = AbsVal(diff)
            
    #         # Expression 2: when x < y, |x - y| = y - x
    #         expected_diff = Sub(y2, x2)
            
    #         # Compute and verify
    #         abs_diff_value = abs_diff.compute()
    #         expected_diff_value = expected_diff.compute()
            
    #         self.assertAlmostEqual(abs_diff_value, expected_diff_value)
            
    #         # Backward pass
    #         abs_diff.backward()
    #         expected_diff.backward()
            
    #         # When x < y, the gradients are:
    #         # d/dx|x-y| = d/dx(-(x-y)) = -1
    #         # d/dy|x-y| = d/dy(-(x-y)) = 1
    #         self.assertEqual(x1.grad, -1.0)
    #         self.assertEqual(y1.grad, 1.0)
    #         self.assertEqual(x2.grad, -1.0)
    #         self.assertEqual(y2.grad, 1.0)


    ########################### test neg with other operators ###########################
    

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main() 