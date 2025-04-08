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
    
    ######## test absval with multiplcation
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
                if abs(variables[i].value) < 1e-6:
                    variables[i] = ElementaryVar(f"x{i}", random.uniform(0.1, 10))

            # Create a copy for the second expression to have independent gradients
            variables2 = [ElementaryVar(f"y{i}", variables[i].value) for i in range(n_vars)]
            
            for i in range(n_vars):
                if abs(variables2[i].value) < 1e-6:
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


    ######## test absval with division
    def test_absval_div(self):
        """
        Test that |x₁*x₂*...*xₙ/y₁*y₂*...*yₙ| behaves exactly like |x₁*x₂*...*xₙ|/|y₁*y₂*...*yₙ|.
        This is a mathematical identity derived from |a/b| = |a|/|b|
        """
        for _ in range(5000):
            # Generate random numbers of numerator and denominator variables
            num_vars = random.randint(2, 10)
            
            # Create variables with mixed signs, avoiding values close to zero for denominator
            numerator_vars = [ElementaryVar(f"x{i}", random.uniform(-10, 10)) for i in range(num_vars)]
            denominator_vars = [ElementaryVar(f"y{i}", random.uniform(-10, -0.5) if random.random() < 0.5 else random.uniform(0.5, 10)) 
                                for i in range(num_vars)] # make sure the denominator is not 0
            
            # make sure the numerator is not too close to zero.
            for i in range(num_vars):
                if abs(numerator_vars[i].value) < 1e-6:
                    numerator_vars[i] = ElementaryVar(f"x{i}", random.uniform(0.1, 10))

            # Create copies for the second expression
            numerator_vars2 = [ElementaryVar(f"x2_{i}", numerator_vars[i].value) for i in range(num_vars)]
            denominator_vars2 = [ElementaryVar(f"y2_{i}", denominator_vars[i].value) for i in range(num_vars)]
            
            # Build the numerator and denominator expressions
            numerator = numerator_vars[0]
            for i in range(1, num_vars):
                numerator = Mult(numerator, numerator_vars[i])
                
            denominator = denominator_vars[0]
            for i in range(1, num_vars):
                denominator = Mult(denominator, denominator_vars[i])
            
            # Expression 1: |numerator/denominator|
            ratio = Div(numerator, denominator)
            abs_ratio = AbsVal(ratio)
            
            # Expression 2: |numerator|/|denominator|
            numerator2 = numerator_vars2[0]
            for i in range(1, num_vars):
                numerator2 = Mult(numerator2, numerator_vars2[i])
                
            denominator2 = denominator_vars2[0]
            for i in range(1, num_vars):
                denominator2 = Mult(denominator2, denominator_vars2[i])
            
            abs_numerator = AbsVal(numerator2)
            abs_denominator = AbsVal(denominator2)
            ratio_of_abs = Div(abs_numerator, abs_denominator)
            
            # Compute and verify both expressions have same value
            abs_ratio_value = abs_ratio.compute()
            ratio_of_abs_value = ratio_of_abs.compute()
            
            self.assertAlmostEqual(abs_ratio_value, ratio_of_abs_value)
            
            # Perform backward pass on both expressions
            abs_ratio.backward()
            ratio_of_abs.backward()
            
            # time to calculate the analytical gradients: as follows: 
            # 1. L = |x_1 * x_2 * ... * x_n |/ |y_1 * y_2 * ... * y_m| = (|x_1| * |x_2| * ... * |x_n|) / (|y_1| * |y_2| * ... * |y_m|)
            # 2. dL / d x_i = |product_but_xi| * sign(x_i) / |y_1 * y_2 * ... * y_m|    
            # 3. dL / d y_i = |product x_i| / (|product_but_yi| * y_i ^ 2) * -sign(y_i)


            # the code above guarantees that x_i is not too close to zero, so we can compute the product all but x_i efficiently
            abs_product_but_xi = [abs_numerator.compute() / abs(numerator_vars[i].value) for i in range(num_vars)]  
            abs_product_but_yi = [abs_denominator.compute() / abs(denominator_vars[i].value) for i in range(num_vars)] 

            # verify the gradients of the numerators 

            for i in range(num_vars):
                expected_grad = (abs_product_but_xi[i] * np.sign(numerator_vars[i].value).item()) / abs_denominator.compute()
                self.assertAlmostEqual(numerator_vars[i].grad, expected_grad)
                self.assertAlmostEqual(numerator_vars2[i].grad, expected_grad)

            for i in range(num_vars):
                expected_grad = (-np.sign(denominator_vars[i].value).item() * abs_numerator.compute()) / (abs_product_but_yi[i] * denominator_vars[i].value ** 2)
                self.assertAlmostEqual(denominator_vars[i].grad, expected_grad)
                self.assertAlmostEqual(denominator_vars2[i].grad, expected_grad)    


    ######## test absval with addition
    def test_absval_add_2_positives(self):
        """
        Test that |x + y| behaves the same as x + y when x and y are guaranteed to be positive.
        When both operands are positive, the absolute value doesn't change anything.
        """
        for _ in range(1000):
            # Generate positive random values
            x_val = random.uniform(0.1, 10)
            y_val = random.uniform(0.1, 10)
            
            x1 = ElementaryVar("x1", x_val)
            y1 = ElementaryVar("y1", y_val)
            
            x2 = ElementaryVar("x2", x_val)
            y2 = ElementaryVar("y2", y_val)
            
            # Expression 1: |x + y|
            sum_expr = Add(x1, y1)
            abs_sum = AbsVal(sum_expr)
            
            # Expression 2: x + y (no absolute value needed since both are positive)
            simple_sum = Add(x2, y2)
            
            # Compute and verify
            abs_sum_value = abs_sum.compute()
            simple_sum_value = simple_sum.compute()
            
            self.assertAlmostEqual(abs_sum_value, simple_sum_value)
            
            # Backward pass
            abs_sum.backward()
            simple_sum.backward()
            
            # For positive values, gradient of |x + y| should be the same as gradient of x + y
            self.assertEqual(x1.grad, x2.grad)
            self.assertEqual(y1.grad, y2.grad)
            
            # Also verify that both gradients are 1 (since d/dx(x+y) = d/dy(x+y) = 1)
            self.assertEqual(x1.grad, 1.0)
            self.assertEqual(y1.grad, 1.0)


    def test_absval_add_pos_neg_sum_pos(self):
        """
        Test that |x + y| behaves the same as x + y when x + y is positive (testing the case where x and y have opposite signs).
        """
        for _ in range(1000):
            x_val = random.uniform(20, 50)
            y_val = random.uniform(-19, -1)
            # this way we are sure that x + y is positive 
            
            x1 = ElementaryVar("x1", x_val)
            y1 = ElementaryVar("y1", y_val)
            
            x2 = ElementaryVar("x2", x_val)
            y2 = ElementaryVar("y2", y_val)
            
            abs_sum = AbsVal(Add(x1, y1))
            # Expression 2: x + y
            simple_sum = Add(x2, y2)
            
            self.assertEqual(abs_sum.value, simple_sum.value)
            
            # Backward pass
            abs_sum.backward()
            simple_sum.backward()

            # For positive values, gradient of |x + y| should be the same as gradient of x + y
            self.assertEqual(x1.grad, x2.grad)
            self.assertEqual(y1.grad, y2.grad)  

            # the gradients are np.sign(x + y) * d(x + y) / dx 
            # the gradients are np.sign(x + y) * d(x + y) / dy 
            self.assertEqual(x1.grad, 1.0)
            self.assertEqual(y1.grad, 1.0)


    def test_absval_add_pos_neg_sum_neg(self):
        """
        Test that |x + y| behaves the same as x + y when x + y is negative (testing the case where x and y have opposite signs).
        """
        for _ in range(1000):
            x_val = random.uniform(1, 15)
            y_val = random.uniform(-50, -20)
            # this way we are sure that x + y is negative 
            
            x1 = ElementaryVar("x1", x_val)
            y1 = ElementaryVar("y1", y_val)
            
            x2 = ElementaryVar("x2", x_val)
            y2 = ElementaryVar("y2", y_val)
            
            abs_sum = AbsVal(Add(x1, y1))
            # Expression 2: -(x + y)
            simple_sum = Neg(Add(x2, y2))
            
            self.assertEqual(abs_sum.value, simple_sum.value)
            
            # Backward pass
            abs_sum.backward()
            simple_sum.backward()

            # For positive values, gradient of |x + y| should be the same as gradient of x + y
            self.assertEqual(x1.grad, x2.grad)
            self.assertEqual(y1.grad, y2.grad)  

            # the gradients are np.sign(x + y) * d(x + y) / dx 
            # the gradients are np.sign(x + y) * d(x + y) / dy 
            self.assertEqual(x1.grad, -1.0)
            self.assertEqual(y1.grad, -1.0)
            
    
    ######## test absval with subtraction
    def test_absval_sub_pos_neg_diff_pos(self):
        """
        Test that |x - y| behaves the same as x - y when x > y.
        """
        for _ in range(1000):
            x_val = random.uniform(20, 50)
            y_val = random.uniform(0, 20)

            x1 = ElementaryVar("x1", x_val)
            y1 = ElementaryVar("y1", y_val)
            
            x2 = ElementaryVar("x2", x_val)
            y2 = ElementaryVar("y2", y_val)

            diff = Sub(x1, y1)
            abs_diff = AbsVal(diff)

            expected_diff = Sub(x2, y2)

            self.assertEqual(abs_diff.value, expected_diff.value)

            abs_diff.backward()
            expected_diff.backward()

            self.assertEqual(x1.grad, 1.0)
            self.assertEqual(y1.grad, -1.0)
            
            self.assertEqual(x2.grad, 1.0)
            self.assertEqual(y2.grad, -1.0) 


    def test_absval_sub_neg_pos_diff_pos(self):
        """
        Test that |x - y| behaves the same as y - x when x < y.
        """
        for _ in range(1000):
            x_val = random.uniform(0, 20)
            y_val = random.uniform(20, 50)


            x1 = ElementaryVar("x1", x_val)
            y1 = ElementaryVar("y1", y_val)
            
            x2 = ElementaryVar("x2", x_val)
            y2 = ElementaryVar("y2", y_val)

            diff = Sub(x1, y1)
            abs_diff = AbsVal(diff)

            expected_diff = Sub(y2, x2)

            self.assertEqual(abs_diff.value, expected_diff.value)

            abs_diff.backward()
            expected_diff.backward()

            self.assertEqual(x1.grad, -1.0)
            self.assertEqual(y1.grad, 1.0)
            
            self.assertEqual(x2.grad, -1.0)
            self.assertEqual(y2.grad, 1.0)
    

    ######## test absval with exponential  
    def test_absval_exp_even(self):
        """
        Test that |x|^n = x^n when n is even.
        When the exponent is even, the sign of the base doesn't matter.
        """
        for _ in range(1000):
            # Generate a random value (including negative values)
            x_val = random.uniform(-10, 10)
            
            # Ensure x is not too close to zero to avoid numerical issues
            if abs(x_val) < 1e-6:
                x_val = 1.0 if random.random() > 0.5 else -1.0
                
            # Generate a random even exponent
            n_val = 2 * random.randint(1, 5)  # 2, 4, 6, 8, 10
            
            # Create variables
            x1 = ElementaryVar("x1", x_val)
            x2 = ElementaryVar("x2", x_val)
            n = ConstantVar("n", n_val)
            
            # Expression 1: |x|^n
            abs_x = AbsVal(x1)
            abs_x_pow_n = Exp(abs_x, n)
            
            # Expression 2: x^n
            x_pow_n = Exp(x2, n)
            
            # Compute and verify both expressions have same value
            abs_x_pow_n_value = abs_x_pow_n.compute()
            x_pow_n_value = x_pow_n.compute()
            
            self.assertAlmostEqual(abs_x_pow_n_value, x_pow_n_value)
            
            # Check that both expressions match the expected mathematical value
            expected_value = x_val ** n_val
            self.assertAlmostEqual(abs_x_pow_n_value, expected_value)
            
            # Backward pass
            abs_x_pow_n.backward()
            x_pow_n.backward()
            
            # Verify gradients: 
            # For |x|^n: d/dx(|x|^n) = n * |x|^(n-1) * sign(x)
            # For x^n: d/dx(x^n) = n * x^(n-1)
            # When n is even, these are equal
            self.assertAlmostEqual(x1.grad, x2.grad)


    # def test_absval_exp_odd(self):
    #     """
    #     Test that |x|^n = |x| * x^(n-1) when n is odd.
    #     This identity helps verify the relationship between absolute value and odd powers.
    #     """
    #     for _ in range(1000):
    #         # Generate a random value (including negative values)
    #         x_val = random.uniform(-10, 10)
            
    #         # Ensure x is not too close to zero to avoid numerical issues
    #         if abs(x_val) < 1e-6:
    #             x_val = 1.0 if random.random() > 0.5 else -1.0
                
    #         # Generate a random odd exponent
    #         n_val = 2 * random.randint(1, 5) + 1  # 3, 5, 7, 9, 11
            
    #         # Create variables
    #         x1 = ElementaryVar("x1", x_val)
    #         x2 = ElementaryVar("x2", x_val)
    #         x3 = ElementaryVar("x3", x_val)
    #         n = ConstantVar("n", n_val)
    #         n_minus_1 = ConstantVar("n-1", n_val - 1)
            
    #         # Expression 1: |x|^n
    #         abs_x_pow_n = Exp(AbsVal(x1), n)
            
    #         # Expression 2: |x| * x^(n-1)
    #         abs_x = AbsVal(x2)
    #         x_pow_n_minus_1 = Exp(x3, n_minus_1)
    #         identity_expr = Mult(abs_x, x_pow_n_minus_1)
            
    #         # Compute and verify both expressions have same value
    #         abs_x_pow_n_value = abs_x_pow_n.compute()
    #         identity_expr_value = identity_expr.compute()
            
    #         self.assertAlmostEqual(abs_x_pow_n_value, identity_expr_value)
            
    #         # Check the expected mathematical value (should be |x|^n)
    #         expected_value = (abs(x_val) ** n_val)
    #         self.assertAlmostEqual(abs_x_pow_n_value, expected_value)
            
    #         # Alternative calculation: |x| * x^(n-1)
    #         alt_expected = abs(x_val) * (x_val ** (n_val - 1))
    #         self.assertAlmostEqual(identity_expr_value, alt_expected)
            
    #         # Backward pass
    #         abs_x_pow_n.backward()
    #         identity_expr.backward()
            


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main() 