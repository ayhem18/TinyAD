"""
This is file to test more complex combinations of the basic binary operators: +, -, *, /
"""

from typing import Callable, Dict, List, Optional, Tuple
import unittest
import random
import numpy as np

from tinyad.autoDiff.common import NUM
from tinyad.autoDiff.operators.binary_ops import Add, Mult, Sub
from tinyad.autoDiff.var import ElementaryVar, ConstantVar, Var


class TestPolynomialExpressions(unittest.TestCase):
    """Test class for combined operations of Add, Mult, Sub, and Div."""
    
    def _create_variables(self, n_vars):
        """Helper to create a list of ElementaryVar objects with random values."""
        return [ElementaryVar(f"x{i}", round(random.uniform(0.5, 2.0), 3)) for i in range(n_vars)]
    

    def _create_polynomial_term(self, variables: List[Var], max_subset_size=5) -> Tuple[Var, Dict[int, int]]:
        """Create a single polynomial term with random variables and powers."""
        n_vars = len(variables)
        subset_size = random.randint(1, min(max_subset_size, n_vars))
        var_indices = random.sample(range(n_vars), subset_size)
        
        # Assign random exponents (powers) to each variable in this term
        exponents = {}
        for idx in var_indices:
            exponents[idx] = random.randint(1, 5)
        
        # Create the product expression for this term
        term = ConstantVar("const", 1.0)
        for idx, power in exponents.items():
            # Multiply by variable raised to power
            for _ in range(power):
                term = Mult(term, variables[idx])
        
        # the exponents variable is a dictionary that maps the variable index (from the original `variables` list)
        # to the power to which the variable is raised in this term
        return term, exponents    

    def _build_expression(self, terms: Optional[List[Var]], start_term: Optional[Var], binaryOp: Callable[[Var, Var], Var]) -> Var:
        """Combine terms into an expression using the specified operation."""
        if not terms:
            return ConstantVar("zero", 0.0)
        
        # make sure the binaryOp is a callable that accepts two arguments
        if not callable(binaryOp):
            raise TypeError("binaryOp must be a callable that accepts two arguments") 
        else:
            # create 2 variables 
            dummyVar1 = ElementaryVar("dummyVar1", 0.0)
            dummyVar2 = ElementaryVar("dummyVar2", 0.0) 
            res = binaryOp(dummyVar1, dummyVar2)
            # check if the result is a Var
            if not isinstance(res, Var):
                raise TypeError("binaryOp must return a Var") 
        
        if start_term is None:
            expression = terms[0]
            for i in range(1, len(terms)):
                expression = binaryOp(expression, terms[i])
        else:
            expression = start_term
            for i in range(len(terms)):
                expression = binaryOp(expression, terms[i])

        return expression

    
    def _calculate_multiplicative_term_value(self, variables: List[Var], exponents: Dict[int, int]) -> NUM:
        """Calculate the value of a single term (know to be the product of variables to some powers)."""
        term_value = 1.0
        for idx, power in exponents.items():
            term_value *= variables[idx].value ** power
        return term_value


    def _calculate_expected_gradient(self, variables: List[Var], 
                                     term_exponents: List[Dict[int, int]], 
                                     term_sign: NUM = 1) -> List[NUM]:
        """Calculate expected gradients for each variable from a list of terms."""
        n_vars = len(variables)
        expected_gradients = [0] * n_vars
        
        for var_idx in range(n_vars):
            for exponents in term_exponents:
                if var_idx in exponents:
                    # For each term containing this variable
                    power = exponents[var_idx]
                    
                    # Calculate term value with the variable's contribution reduced by 1 power
                    term_coefficient = 1.0
                    for other_idx, other_power in exponents.items():
                        if other_idx == var_idx:
                            # Differentiate with respect to this variable
                            term_coefficient *= power * (variables[var_idx].value ** (power - 1))
                        else:
                            # For other variables, use the full power
                            term_coefficient *= variables[other_idx].value ** other_power
                    
                    expected_gradients[var_idx] += term_sign * term_coefficient
        
        return expected_gradients
    

    def _verify_gradients(self, variables: List[Var], expected_gradients: List[NUM], counts: List[NUM]):
        """Verify that the actual gradients match expected gradients."""
        for i, var in enumerate(variables):
            if counts[i] == 0:
                self.assertTrue(
                    var.grad is None,
                    f"Variable {i} wasn't used but has gradient {var.grad}"
                )

            else:
                self.assertAlmostEqual(
                    var.grad, 
                    expected_gradients[i],
                    msg=f"Variable {i} gradient incorrect. Expected {expected_gradients[i]}, got {var.grad}"
                )

    
    def test_poly_add_mult(self):
        """
        Test complex polynomial expressions and their gradients.
        A polynomial is represented as a sum of products of variables raised to powers.
        """
        for _ in range(1000):
            # 1. Generate random variables and terms
            n_vars = random.randint(10, 20)
            n_groups = random.randint(4, 10)
            
            variables = self._create_variables(n_vars)
            
            # 2 & 3. Create terms with random exponents
            terms = []
            term_exponents = []
            
            for _ in range(n_groups):
                term, exponents = self._create_polynomial_term(variables)
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
            expected_gradients = self._calculate_expected_gradient(variables, term_exponents)

            # count how many times each variable is used in the terms
            counts = [0] * n_vars 
            for exponents in term_exponents:
                for idx in exponents:
                    counts[idx] += 1

            self._verify_gradients(variables, expected_gradients, counts)


    def test_poly_add_mult_sub(self):
        """
        Test complex polynomial expressions using both Add and Sub operations.
        Creates two polynomials - one with Add and one with Sub - and tests their sum.
        """
        for _ in range(1000):
            # 1. Generate random variables
            n_vars = random.randint(10, 20)
            variables = self._create_variables(n_vars)
            
            # 2. Create terms for two polynomials
            n_terms_p1 = random.randint(3, 8)
            n_terms_p2 = random.randint(3, 8)
            
            # Terms and exponents for polynomial P1 (using Add)
            p1_terms = []
            p1_term_exponents = []
            
            for _ in range(n_terms_p1):
                term, exponents = self._create_polynomial_term(variables)
                p1_terms.append(term)
                p1_term_exponents.append(exponents)
            
            # Terms and exponents for polynomial P2 (using Sub)
            p2_terms = []
            p2_term_exponents = []
            
            for _ in range(n_terms_p2):
                term, exponents = self._create_polynomial_term(variables)
                p2_terms.append(term)
                p2_term_exponents.append(exponents)
            
            # 3. Build the polynomials
            p1 = self._build_expression(p1_terms, None, Add)
            p2 = self._build_expression(p2_terms, ConstantVar("const", 0), Sub)
            
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
            expected_gradients_p1 = self._calculate_expected_gradient(variables, p1_term_exponents)
                        
            expected_gradients_p2 = self._calculate_expected_gradient(
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
            self._verify_gradients(variables, expected_gradients, counts)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    unittest.main()
    