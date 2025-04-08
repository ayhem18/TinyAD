"""
This file tests the exponentiation operator in combination with other operations
"""

import unittest
import random
import numpy as np

from tinyad.autoDiff.operators.binary_ops import Add, Mult, Sub, Div, Exp
from tinyad.autoDiff.var import ElementaryVar, ConstantVar
from tinyad.tests.operators.combined_ops_tests.binary_operators.test_bin_ops_base_test import BinaryOperatorsBaseTest


class TestPolynomialExpressesionsExponentiation(BinaryOperatorsBaseTest):
    """Test class for polynomial expressions with exponentiation."""
    
    def test_poly_with_exp(self):
        """
        Test complex polynomial expressions using the Exp operator.
        A polynomial is represented as a sum of products of variables raised to powers.
        """
        for _ in range(500):
            # 1. Generate random variables and terms
            n_vars = random.randint(5, 15)
            n_groups = random.randint(3, 8)
            
            variables = super()._create_variables(n_vars)
            
            # 2 & 3. Create terms with random float exponents using Exp
            terms = []
            term_exponents = []
            
            for _ in range(n_groups):
                term, exponents = super()._create_multiplicative_term_float_expos(variables)
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
            
            # 7. Verify gradients using the float exponent gradient calculator
            expected_gradients = self._calculate_gradient_multiplicative_term(variables, term_exponents)

            # Count how many times each variable is used in the terms
            counts = [0] * n_vars 
            for exponents in term_exponents:
                for idx in exponents:
                    counts[idx] += 1

            super()._verify_gradients(variables, expected_gradients, counts)
    
    def test_poly_add_exp_sub(self):
        """
        Test complex polynomial expressions using both Add and Sub operations with Exp.
        Creates two polynomials - one with Add and one with Sub - and tests their sum.
        """
        for _ in range(500):
            # 1. Generate random variables
            n_vars = random.randint(5, 15)
            variables = super()._create_variables(n_vars)
            
            # 2. Create terms for two polynomials
            n_terms_p1 = random.randint(2, 6)
            n_terms_p2 = random.randint(2, 6)
            
            # Terms and exponents for polynomial P1 (using Add)
            p1_terms = []
            p1_term_exponents = []
            
            for _ in range(n_terms_p1):
                term, exponents = super()._create_multiplicative_term_float_expos(variables)
                p1_terms.append(term)
                p1_term_exponents.append(exponents)
            
            # Terms and exponents for polynomial P2 (using Sub)
            p2_terms = []
            p2_term_exponents = []
            
            for _ in range(n_terms_p2):
                term, exponents = super()._create_multiplicative_term_float_expos(variables)
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
            
            # Expected value for p2: negation of the sum
            expected_p2 = -sum(self._calculate_multiplicative_term_value(variables, exponents) 
                               for exponents in p2_term_exponents)
            
            expected_result = expected_p1 + expected_p2
            
            self.assertAlmostEqual(result, expected_result)
            
            # 6. Perform backward pass
            final.backward()
            
            # 7. Calculate expected gradients for each polynomial
            expected_gradients_p1 = super()._calculate_gradient_multiplicative_term(
                variables, p1_term_exponents)
            
            expected_gradients_p2 = super()._calculate_gradient_multiplicative_term(
                variables, p2_term_exponents, term_sign=-1)
            
            # Combined expected gradients
            expected_gradients = [expected_gradients_p1[i] + expected_gradients_p2[i] 
                                 for i in range(n_vars)]

            # Count which variables were used
            counts = [0] * n_vars 
            for exponents in p1_term_exponents + p2_term_exponents:
                for idx in exponents:
                    counts[idx] += 1

            # Verify the gradients
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
            term1, term1_coeffs_powers = super()._create_additive_exponential_term_float_expos(variables)
            term2, term2_coeffs_powers = super()._create_additive_exponential_term_float_expos(variables)
            
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



class TestRationalExpressionsExponentiation(BinaryOperatorsBaseTest):
    """Test class for rational expressions with exponentiation."""
    
    def test_rational_expression_with_exp(self):
        """
        Test rational expressions P(x)/Q(x) using Exp operator where:
        P(x) = sum_i (a_i * x_i^p_i)
        Q(x) = sum_i (b_i * x_i^q_i)
        
        Verifies both computation and gradient calculation.
        """
        for _ in range(500):
            # 1. Generate random variables (avoid values close to zero)
            n_vars = random.randint(3, 8)
            variables = [ElementaryVar(f"x{i}", round(random.uniform(0.5, 2.0), 2)) for i in range(n_vars)]
            
            # 2. Create additive terms using variables with float exponents
            # For numerator P(x)
            numerator_terms = []
            numerator_term_exponents = []
            
            for _ in range(random.randint(2, 5)):
                term, exponents = super()._create_multiplicative_term_float_expos(variables, max_subset_size=3)
                numerator_terms.append(term)
                numerator_term_exponents.append(exponents)
            
            # For denominator Q(x)
            denominator_terms = []
            denominator_term_exponents = []
            
            for _ in range(random.randint(2, 5)):
                term, exponents = super()._create_multiplicative_term_float_expos(variables, max_subset_size=3)
                denominator_terms.append(term)
                denominator_term_exponents.append(exponents)
            
            # Build the numerator and denominator
            numerator = super()._build_expression(numerator_terms, None, Add)
            denominator = super()._build_expression(denominator_terms, None, Add)
            
            # Add constant to denominator to avoid division by zero
            constant_term = ConstantVar("const", random.uniform(1.0, 2.0))
            denominator = Add(denominator, constant_term)
            
            # 3. Create the rational expression P(x)/Q(x)
            rational_expr = Div(numerator, denominator)
            
            # 4. Compute and verify the value
            result = rational_expr.compute()
            
            # Calculate expected values separately
            numerator_value = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                                 for exponents in numerator_term_exponents)
            
            denominator_value = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                                    for exponents in denominator_term_exponents) + constant_term.value
            
            expected_result = numerator_value / denominator_value
            
            self.assertAlmostEqual(result, expected_result)
            
            # 5. Perform backward pass
            rational_expr.backward()
            
            # 6. Calculate expected gradients using the quotient rule:
            # d/dx[P(x)/Q(x)] = (Q(x)*dP/dx - P(x)*dQ/dx) / Q(x)²
            numerator_gradients = self._calculate_gradient_multiplicative_term(
                variables, numerator_term_exponents)
            
            denominator_gradients = self._calculate_gradient_multiplicative_term(
                variables, denominator_term_exponents)
            
            # Apply quotient rule to each variable's gradient
            expected_gradients = [0] * n_vars
            for idx in range(n_vars):
                if numerator_gradients[idx] != 0 or denominator_gradients[idx] != 0:
                    expected_gradients[idx] = (denominator_value * numerator_gradients[idx] - 
                                             numerator_value * denominator_gradients[idx]) / (denominator_value ** 2)
            
            # 7. Determine which variables were used
            counts = [0] * n_vars
            for exponents in numerator_term_exponents + denominator_term_exponents:
                for idx in exponents:
                    counts[idx] += 1
            
            # 8. Verify gradients
            super()._verify_gradients(variables, expected_gradients, counts)
    
    def test_complex_rational_expression_with_exp(self):
        """
        Test more complex rational expressions of the form (P₁+P₂)/(Q₁*Q₂) with Exp.
        This tests nesting of operations in both numerator and denominator.
        """
        for _ in range(500):
            # 1. Generate random variables (avoid values close to zero)
            n_vars = random.randint(3, 8)
            variables = [ElementaryVar(f"x{i}", round(random.uniform(0.5, 3.0), 3)) for i in range(n_vars)]
            
            # 2. Create components of the expression using Exp
            # For numerator components: P₁ and P₂
            p1_terms = []
            p1_term_exponents = []
            
            for _ in range(random.randint(1, 3)):
                term, exponents = super()._create_multiplicative_term_float_expos(variables, max_subset_size=2)
                p1_terms.append(term)
                p1_term_exponents.append(exponents)
            
            p2_terms = []
            p2_term_exponents = []
            
            for _ in range(random.randint(1, 3)):
                term, exponents = super()._create_multiplicative_term_float_expos(variables, max_subset_size=2)
                p2_terms.append(term)
                p2_term_exponents.append(exponents)
            
            # For denominator components: Q₁ and Q₂
            q1_terms = []
            q1_term_exponents = []
            
            for _ in range(random.randint(1, 3)):
                term, exponents = super()._create_multiplicative_term_float_expos(variables, max_subset_size=2)
                q1_terms.append(term)
                q1_term_exponents.append(exponents)
            
            q2_terms = []
            q2_term_exponents = []
            
            for _ in range(random.randint(1, 3)):
                term, exponents = super()._create_multiplicative_term_float_expos(variables, max_subset_size=2)
                q2_terms.append(term)
                q2_term_exponents.append(exponents)
            
            # Build P₁, P₂, Q₁, and Q₂
            p1 = super()._build_expression(p1_terms, None, Add)
            p2 = super()._build_expression(p2_terms, None, Add)
            q1 = super()._build_expression(q1_terms, None, Add)
            q2 = super()._build_expression(q2_terms, None, Add)
            
            # Add constants to denominators to avoid division by zero
            const1 = ConstantVar("c1", 1.0)
            const2 = ConstantVar("c2", 1.0)
            q1 = Add(q1, const1)
            q2 = Add(q2, const2)
            
            # 3. Create the complex rational expression (P₁ + P₂) / (Q₁ * Q₂)
            numerator = Add(p1, p2)
            denominator = Mult(q1, q2)
            rational_expr = Div(numerator, denominator)
            
            # 4. Compute the result
            result = rational_expr.compute()
            
            # Calculate expected values
            p1_value = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                          for exponents in p1_term_exponents)
            
            p2_value = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                          for exponents in p2_term_exponents)
            
            q1_value = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                          for exponents in q1_term_exponents) + const1.value
            
            q2_value = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                          for exponents in q2_term_exponents) + const2.value
            
            expected_result = (p1_value + p2_value) / (q1_value * q2_value)
            self.assertAlmostEqual(result, expected_result)
            
            # 5. Perform backward pass
            rational_expr.backward()
            
            # 6. Verify gradients with finite differences - this is complex enough to warrant numerical approximation
            epsilon = 1e-6
            numerical_gradients = []
            
            for idx in range(n_vars):
                # Save original value
                original = variables[idx].value
                
                # Compute f(x+ε)
                variables[idx].value = original + epsilon
                # Recalculate the values with the perturbed variable
                p1_plus = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                             for exponents in p1_term_exponents)
                
                p2_plus = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                             for exponents in p2_term_exponents)
                
                q1_plus = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                             for exponents in q1_term_exponents) + const1.value
                
                q2_plus = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                             for exponents in q2_term_exponents) + const2.value
                
                result_plus = (p1_plus + p2_plus) / (q1_plus * q2_plus)
                
                # Compute f(x-ε)
                variables[idx].value = original - epsilon
                p1_minus = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                              for exponents in p1_term_exponents)
                
                p2_minus = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                              for exponents in p2_term_exponents)
                
                q1_minus = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                              for exponents in q1_term_exponents) + const1.value
                
                q2_minus = sum(self._calculate_multiplicative_term_value(variables, exponents) 
                              for exponents in q2_term_exponents) + const2.value
                
                result_minus = (p1_minus + p2_minus) / (q1_minus * q2_minus)
                
                # Restore original value
                variables[idx].value = original
                
                # Central difference approximation: (f(x+ε) - f(x-ε)) / 2ε
                numerical_grad = (result_plus - result_minus) / (2 * epsilon)
                numerical_gradients.append(numerical_grad)
            
            # 7. Count which variables were used
            counts = [0] * n_vars
            for exponents in p1_term_exponents + p2_term_exponents + q1_term_exponents + q2_term_exponents:
                for idx in exponents:
                    counts[idx] += 1
            
            # 8. Compare autodiff gradients with numerical gradients
            for idx in range(n_vars):
                if counts[idx] > 0:  # Skip variables not used in expression
                    self.assertAlmostEqual(
                        variables[idx].grad,
                        numerical_gradients[idx],
                        delta=1e-4  # Allow some small error due to numerical differentiation
                    )


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main() 