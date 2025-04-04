"""
This is file to test more complex combinations of the basic binary operators: +, -, *, /
"""

import unittest
import random
import numpy as np

from tinyad.autoDiff.operators.binary_ops import Add, Mult, Sub, Div
from tinyad.autoDiff.var import ElementaryVar, ConstantVar
from tinyad.tests.operators.combined_ops_tests.base_test import BaseTest


class TestBasicCombinations(BaseTest):
    """Test class for combined operations of Add, Mult, Sub."""

    def test_mult_as_add(self):
        """Test the multiplication of two variables as an addition of two variables."""
        for _ in range(5000):            
            val = round(random.uniform(1, 10), 3)
            
            var1 = ElementaryVar("x", val)
            var2 = ElementaryVar("y", val)
            
            n = random.randint(2, 100)

            resAdd = ConstantVar("res", 0)
            
            for i in range(n):
                resAdd = Add(resAdd, var1)
            
            resMult = Mult(var2, ConstantVar("const", n))

            resAdd.compute()
            resMult.compute()

            self.assertEqual(var1.value, val)
            self.assertEqual(var2.value, val)

            self.assertAlmostEqual(resAdd.value, val * n)
            self.assertAlmostEqual(resMult.value, val * n)
            
            resAdd.backward()
            resMult.backward()

            self.assertEqual(var1.grad, n)
            self.assertEqual(var2.grad, n)

    
    def test_mult_as_sub(self):
        """Test the multiplication of two variables as a subtraction of two variables."""
        for _ in range(5000):
            val = round(random.uniform(1, 10), 3)
            
            var1 = ElementaryVar("x", val)
            var2 = ElementaryVar("y", val)

            n = random.randint(2, 100)

            resSub = ConstantVar("res", 0)
            for i in range(n):
                resSub = Sub(resSub, var1)
            
            resMult = Mult(var2, ConstantVar("const", -n))

            resSub.compute()
            resMult.compute()

            self.assertEqual(var1.value, val)
            self.assertEqual(var2.value, val)

            self.assertAlmostEqual(resSub.value, -val * n)
            self.assertAlmostEqual(resMult.value, -val * n)

            resSub.backward()
            resMult.backward()

            self.assertEqual(var1.grad, -n)
            self.assertEqual(var2.grad, -n)


    def test_same_powers_divided(self):
        """
        Test division of the same variable raised to the same power.
        Tests the expression x^n / x^n = 1 and verifies gradient is zero.
        """
        for _ in range(1000):
            # Avoid values too close to zero to prevent numerical instability
            val = round(random.uniform(0.5, 10.0), 3)
            var = ElementaryVar("x", val)
            
            # Choose a random exponent
            n = random.randint(1, 5)
            
            # Create x^n for numerator
            numerator = var
            for _ in range(n-1):
                numerator = Mult(numerator, var)
            
            # Create x^n for denominator
            denominator = var
            for _ in range(n-1):
                denominator = Mult(denominator, var)
            
            # Create division expression: x^n / x^n
            division = Div(numerator, denominator)
            
            # Compute result - should be 1
            result = division.compute()
            self.assertAlmostEqual(result, 1.0)
            
            # Perform backward pass - gradient should be 0
            division.backward()
            self.assertAlmostEqual(var.grad, 0.0)  # Use places to handle potential floating point issues

    def test_diff_powers_divided(self):
        """
        Test division of the same variable raised to different powers.
        Tests the expression x^n1 / x^n2 = x^(n1-n2) and verifies gradient.
        """
        for _ in range(1000):
            # Avoid values too close to zero to prevent numerical instability
            val = round(random.uniform(0.5, 10.0), 3)
            var = ElementaryVar("x", val)
            
            # Choose two different exponents
            n1 = random.randint(2, 6)
            n2 = random.randint(1, n1-1)  # Ensure n1 > n2 to avoid negative powers
            
            # Create x^n1 for numerator
            numerator = var
            for _ in range(n1-1):
                numerator = Mult(numerator, var)
            
            # Create x^n2 for denominator
            denominator = var
            for _ in range(n2-1):
                denominator = Mult(denominator, var)
            
            # Create division expression: x^n1 / x^n2
            division = Div(numerator, denominator)
            
            # Compute result - should be x^(n1-n2)
            result = division.compute()
            expected_result = val ** (n1 - n2)
            self.assertAlmostEqual(result, expected_result)
            
            # Perform backward pass
            division.backward()
            
            # Calculate expected gradient: d/dx(x^(n1-n2)) = (n1-n2) * x^(n1-n2-1)
            power_diff = n1 - n2
            expected_gradient = power_diff * (val ** (power_diff - 1))
            
            self.assertAlmostEqual(var.grad, expected_gradient)


class TestPolynomialExpressions(BaseTest):
    """Test class for combined operations of Add, Mult, Sub, and Div."""    
    
    def test_poly_add_mult(self):
        """
        Test complex polynomial expressions and their gradients.
        A polynomial is represented as a sum of products of variables raised to powers.
        """
        for _ in range(5000):
            # 1. Generate random variables and terms
            n_vars = random.randint(10, 20)
            n_groups = random.randint(4, 10)
            
            variables = super()._create_variables(n_vars)
            
            # 2 & 3. Create terms with random exponents
            terms = []
            term_exponents = []
            
            for _ in range(n_groups):
                term, exponents = super()._create_multiplicative_term(variables)
                terms.append(term)
                term_exponents.append(exponents)
            
            # 4. Build the polynomial by adding terms
            polynomial = self._build_expression(terms, None, Add)
            
            # 5. Compute and verify the polynomial value
            result = polynomial.compute()
            
            expected_result = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                                 for exponents in term_exponents)
            
            self.assertAlmostEqual(result, expected_result)
            
            # 6. Perform backward pass
            polynomial.backward()
            
            # 7. Verify gradients
            expected_gradients = self._calculate_gradient_multiplicative_term(variables, term_exponents)

            # count how many times each variable is used in the terms
            counts = [0] * n_vars 
            for exponents in term_exponents:
                for idx in exponents:
                    counts[idx] += 1

            super()._verify_gradients(variables, expected_gradients, counts)


    def test_poly_add_mult_sub(self):
        """
        Test complex polynomial expressions using both Add and Sub operations.
        Creates two polynomials - one with Add and one with Sub - and tests their sum.
        """
        for _ in range(5000):
            # 1. Generate random variables
            n_vars = random.randint(10, 20)
            variables = super()._create_variables(n_vars)
            
            # 2. Create terms for two polynomials
            n_terms_p1 = random.randint(3, 8)
            n_terms_p2 = random.randint(3, 8)
            
            # Terms and exponents for polynomial P1 (using Add)
            p1_terms = []
            p1_term_exponents = []
            
            for _ in range(n_terms_p1):
                term, exponents = super()._create_multiplicative_term(variables)
                p1_terms.append(term)
                p1_term_exponents.append(exponents)
            
            # Terms and exponents for polynomial P2 (using Sub)
            p2_terms = []
            p2_term_exponents = []
            
            for _ in range(n_terms_p2):
                term, exponents = super()._create_multiplicative_term(variables)
                p2_terms.append(term)
                p2_term_exponents.append(exponents)
            
            # 3. Build the polynomials
            p1 = super()._build_expression(p1_terms, None, Add)
            p2 = super()._build_expression(p2_terms, ConstantVar("const", 0), Sub)
            
            # 4. Combine the polynomials: final = p1 + p2
            final = Add(p1, p2)
            
            # 5. Compute and verify the result
            result = final.compute()
            
            # Expected value for p1: sum of all its terms
            expected_p1 = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                             for exponents in p1_term_exponents)
            
            # Expected value for p2: first term minus all others
            expected_p2 = 0

            expected_p2 = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                                for exponents in p2_term_exponents)
            
            expected_result = expected_p1 - expected_p2
            
            self.assertAlmostEqual(result, expected_result)
            
            # 6. Backward pass
            final.backward()
            
            # 7. Calculate expected gradients
            expected_gradients_p1 = super()._calculate_gradient_multiplicative_term(variables, p1_term_exponents)
                        
            expected_gradients_p2 = super()._calculate_gradient_multiplicative_term(
                variables, p2_term_exponents, term_sign=-1)
            
            
            # Combined expected gradients
            expected_gradients = [expected_gradients_p1[i] + expected_gradients_p2[i] 
                                 for i in range(n_vars)]

            counts = [0] * n_vars 
            
            for exponents in p1_term_exponents:
                for idx in exponents:
                    counts[idx] += 1

            for exponents in p2_term_exponents:
                for idx in exponents:
                    counts[idx] += 1

            # Verify the gradients
            super()._verify_gradients(variables, expected_gradients, counts)


    def test_multiply_additive_terms(self):
        """
        Test the multiplication of two additive terms (linear combinations).
        Tests the expression (a₁x₁ + a₂x₂ + ... + aₙxₙ) * (b₁y₁ + b₂y₂ + ... + bₘyₘ)
        and verifies gradient computation.
        """
        for _ in range(5000):
            # 1. Generate random variables
            n_vars = random.randint(5, 15)
            variables = super()._create_variables(n_vars)
            
            subset_size = random.randint(2, n_vars)

            # 2. Create two additive terms
            term1, term1_coeffs = super()._create_additive_term(variables, max_subset_size=subset_size)
            term2, term2_coeffs = super()._create_additive_term(variables, max_subset_size=subset_size)
            
            # 4. Multiply the terms
            product = Mult(term1, term2)
            
            # 5. Compute and verify the value
            result = product.compute()
            
            # Calculate expected product value
            term1_value = super()._calculate_additive_term_value(variables, term1_coeffs)
            term2_value = super()._calculate_additive_term_value(variables, term2_coeffs)
            expected_result = term1_value * term2_value
            
            self.assertAlmostEqual(result, expected_result)
            
            # 6. Perform backward pass
            product.backward()
            
            # 7. Calculate expected gradients
            expected_gradients = super()._calculate_gradient_multiplication_two_additive_terms(
                variables, term1_coeffs, term2_coeffs)
            
            # 8. Verify gradients
            # Count which variables were used in either term
            counts = [0] * n_vars
            for idx in set(list(term1_coeffs.keys()) + list(term2_coeffs.keys())):
                counts[idx] = 1
            
            super()._verify_gradients(variables, expected_gradients, counts)


    def test_multiply_additive_exponential_terms(self):
        """
        Test the multiplication of two additive-exponential terms.
        Tests expressions of the form: 
        (a₁x₁^p₁ + a₂x₂^p₂ + ... + aₙxₙ^pₙ) * (b₁x₁^q₁ + b₂x₂^q₂ + ... + bₘxₘ^qₘ)
        where each variable can have a different power in each term.
        """
        for _ in range(1000):
            # 1. Generate random variables
            n_vars = random.randint(4, 10)
            variables = super()._create_variables(n_vars)
            
            # 2. Create two additive-exponential terms
            term1, term1_coeffs_powers = super()._create_additive_exponential_term(variables)
            term2, term2_coeffs_powers = super()._create_additive_exponential_term(variables)
            
            # 3. Multiply the terms
            product = Mult(term1, term2)
            
            # 4. Compute and verify the value
            result = product.compute()
            
            # Calculate expected product value
            term1_value = super()._calculate_additive_exponential_term_value(variables, term1_coeffs_powers)
            term2_value = super()._calculate_additive_exponential_term_value(variables, term2_coeffs_powers)
            expected_result = term1_value * term2_value
            
            self.assertAlmostEqual(result, expected_result)
            
            # 5. Perform backward pass
            product.backward()
            
            # 6. Calculate expected gradients
            expected_gradients = super()._calculate_gradient_multiplication_two_additive_exponential_terms(
                variables, term1_coeffs_powers, term2_coeffs_powers)
            
            # 7. Determine which variables were used in either term
            counts = [0] * n_vars
            for idx in set(list(term1_coeffs_powers.keys()) + list(term2_coeffs_powers.keys())):
                counts[idx] = 1
            
            # 8. Verify gradients
            super()._verify_gradients(variables, expected_gradients, counts)


class TestRationalExpressions(BaseTest):
    """Test class for rational expressions combining all operations."""
    
    def test_rational_expression(self):
        """
        Test rational expressions of the form P(x)/Q(x) where:
        P(x) = sum_i (a_i * x_i^p_i)
        Q(x) = sum_i (b_i * x_i^q_i)
        
        Verifies both computation and gradient calculation.
        """
        for _ in range(5000):
            # 1. Generate random variables (avoid values close to zero)
            n_vars = random.randint(3, 8)
            variables = [ElementaryVar(f"x{i}", round(random.uniform(0.5, 3.0), 3)) for i in range(n_vars)]
            
            # 2. Create additive-exponential terms for numerator and denominator
            # For numerator P(x)
            numerator, numerator_coeffs_powers = self._create_additive_exponential_term(variables)
            
            # For denominator Q(x) - ensure it's non-zero by adding a constant term
            denominator, denominator_coeffs_powers = self._create_additive_exponential_term(variables)
            # Add constant to denominator to avoid division by zero
            constant_term = ConstantVar("const", random.uniform(1.0, 2.0))
            denominator = Add(denominator, constant_term)
            
            # 3. Create the rational expression P(x)/Q(x)
            rational_expr = Div(numerator, denominator)
            
            # 4. Compute and verify the value
            result = rational_expr.compute()
            
            # Calculate expected values separately
            numerator_value = self._calculate_additive_exponential_term_value(variables, numerator_coeffs_powers)
            # For denominator, add the constant term
            denominator_value = self._calculate_additive_exponential_term_value(variables, denominator_coeffs_powers) + constant_term.value
            expected_result = numerator_value / denominator_value
            
            self.assertAlmostEqual(result, expected_result)
            
            # 5. Perform backward pass
            rational_expr.backward()
            
            # 6. Calculate expected gradients using the quotient rule:
            # d/dx[P(x)/Q(x)] = (Q(x)*dP/dx - P(x)*dQ/dx) / Q(x)²
            expected_gradients = [0] * n_vars
            
            for idx in range(n_vars):
                # Initialize derivatives
                dP_dx = 0
                dQ_dx = 0
                
                # Calculate dP/dx_idx if variable appears in numerator
                if idx in numerator_coeffs_powers:
                    coeff, power = numerator_coeffs_powers[idx]
                    if power > 0:  # Only variables with positive power have non-zero gradient
                        dP_dx = coeff * power * (variables[idx].value ** (power - 1))
                
                # Calculate dQ/dx_idx if variable appears in denominator
                if idx in denominator_coeffs_powers:
                    coeff, power = denominator_coeffs_powers[idx]
                    if power > 0:  # Only variables with positive power have non-zero gradient
                        dQ_dx = coeff * power * (variables[idx].value ** (power - 1))
                
                # Apply quotient rule: (Q*dP/dx - P*dQ/dx) / Q²
                if dP_dx != 0 or dQ_dx != 0:  # Only calculate if gradient is non-zero
                    expected_gradients[idx] = (denominator_value * dP_dx - numerator_value * dQ_dx) / (denominator_value ** 2)
            
            # 7. Determine which variables were used in either term
            counts = [0] * n_vars
            var_set = set(list(numerator_coeffs_powers.keys()) + list(denominator_coeffs_powers.keys()))
            for idx in var_set:
                counts[idx] = 1
            
            # 8. Verify gradients
            self._verify_gradients(variables, expected_gradients, counts)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main()
    