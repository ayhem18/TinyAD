"""
This file tests complex combinations of unary operators (AbsVal, Neg) with 
binary operators (Add, Sub, Mult, Div, Exp).
"""

import random
import unittest
import numpy as np

from tinyad.autoDiff.operators.binary_ops import Mult, Div
from tinyad.autoDiff.operators.unary_ops import AbsVal
from tinyad.tests.operators.combined_ops_tests.binary_operators.test_bin_ops_base_test import BinaryOperatorsBaseTest


class TestComplexUniaryBinaryCombinations(BinaryOperatorsBaseTest):
    """Test class for complex combinations of unary and binary operators."""

    def test_abs_polynomial_product(self):
        """
        Test a the product of absolute values of 2 polynomials.
        Tests expressions of the form :  |a_i * x^i + a_{i-1} * x^{i-1} + ... + a_0| * |b_j * x^j + b_{j-1} * x^{j-1} + ... + b_0|
        """
        for _ in range(5000):
            # 1. Generate random variables
            n_vars = random.randint(4, 10)
            variables = super()._create_variables(n_vars)
            
            # 2. Create two additive-exponential terms
            term1, term1_coeffs_powers = super()._create_additive_exponential_term_float_expos(variables)
            term2, term2_coeffs_powers = super()._create_additive_exponential_term_float_expos(variables)
            
            # apply the absolute value to the terms
            term1 = AbsVal(term1)
            term2 = AbsVal(term2)

            # 3. Multiply the terms
            product = Mult(term1, term2)
            
            # 4. Compute and verify the value
            result = product.compute()
            
            # Calculate expected product value
            term1_value = super()._calculate_additive_exponential_term_value(variables, term1_coeffs_powers)
            term2_value = super()._calculate_additive_exponential_term_value(variables, term2_coeffs_powers)
            expected_result = abs(term1_value * term2_value)
            
            self.assertAlmostEqual(result, expected_result)
            
            # 5. Perform backward pass
            product.backward()
            
            # 6. Calculate expected gradients
            expected_gradients = super()._calculate_grad_mult_2_abs_additive_exp_terms(
                variables, term1_coeffs_powers, term2_coeffs_powers)
            
            # 7. Determine which variables were used in either term
            counts = [0] * n_vars
            for idx in set(list(term1_coeffs_powers.keys()) + list(term2_coeffs_powers.keys())):
                counts[idx] = 1
            
            # 8. Verify gradients
            super()._verify_gradients(variables, expected_gradients, counts)


    def test_ratio_2_diff_abs_terms(self):
        """
        This method tests the following expressions: 
        |a_i * x_i ^ p_i + a_{i - 1} * x_{i - 1} ^ p_{i - 1} + ... + a_0| / (|b_i * x_i ^ p_i| + |b_{i - 1} * x_{i - 1} ^ p_{i - 1}| + ... + |b_0|)
        """
        for _ in range(5000):
            # 1. generate the variables
            n_vars = random.randint(4, 10)
            variables = super()._create_variables(n_vars)

            # 2. create the terms
            numerator, numerator_coeffs_powers = super()._create_additive_exponential_term_float_expos(variables)
            denominator, denominator_coeffs_powers = super()._create_additive_abs_exponential_term(variables)
            
            numerator = AbsVal(numerator)

            # 3. create the ratio
            ratio = Div(numerator, denominator)
            
            # 4. compute the value
            ratio_var_value = ratio.compute()

            # compute the expected value
            numerator_value_no_abs = super()._calculate_additive_exponential_term_value(variables, numerator_coeffs_powers)

            denominator_value = super()._calculate_abs_additive_exponential_term_value(variables, denominator_coeffs_powers)
            expected_result = abs(numerator_value_no_abs) / denominator_value

            # 5. verify the value
            self.assertAlmostEqual(ratio_var_value, expected_result)
            
            # 6. perform the backward pass
            ratio.backward()

            # 7. compute the analytical gradients 
            # the gradient is a little bit involved here, so let's get it to it
            # so we have f = |a_i * x_i^p_i + a_{i-1} * x_{i-1}^p_{i-1} + ... + a_0|
            # df / dxi = np.sign(a_i * x_i^p_i + a_{i-1} * x_{i-1}^p_{i-1} + ... + a_0) * p_i * a_i * x_i^(p_i - 1)
            # let g = |b_i * x_i^p_i| + |b_{i-1} * x_{i-1}^p_{i-1}| + ... + |b_0| 
            # dg / dxi = np.sign(b_i * x_i^p_i) * p_i * x_i ^ (p_i - 1) 

            # L = f / g 
            # dL / dxi = (df / dxi * g - f * dg / dxi) / g^2 

            expected_gradients = [0] * n_vars

            for idx in range(n_vars):                
                f = abs(numerator_value_no_abs)
                g = denominator_value 

                df_dxi = 0
                dg_dxi = 0

                # Check if variable appears in first term
                if idx in numerator_coeffs_powers:
                    coeff1, power1 = numerator_coeffs_powers[idx]

                    # df / dxi = np.sign(a_i * x_i^p_i + a_{i-1} * x_{i-1}^p_{i-1} + ... + a_0) * p_i * a_i * x_i^(p_i - 1)
                    df_dxi = np.sign(numerator_value_no_abs).item() * power1 * coeff1 * (variables[idx].value ** (power1 - 1)) 
                
                # Check if variable appears in second term
                if idx in denominator_coeffs_powers:
                    coeff2, power2 = denominator_coeffs_powers[idx]
                    dg_dxi = np.sign(coeff2 * variables[idx].value ** power2).item() * power2 * coeff2 * (variables[idx].value ** (power2 - 1))  

                expected_gradients[idx] = (df_dxi * g - f * dg_dxi) / (g ** 2)

            # 8. verify the gradients   
            counts = [0] * n_vars
            for idx in set(list(numerator_coeffs_powers.keys()) + list(denominator_coeffs_powers.keys())):
                counts[idx] = 1
            super()._verify_gradients(variables, expected_gradients, counts)



if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main() 