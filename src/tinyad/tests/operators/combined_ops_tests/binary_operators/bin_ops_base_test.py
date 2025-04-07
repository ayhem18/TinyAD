import random
from typing import Callable, Dict, List, Optional, Tuple
import unittest

from tinyad.autoDiff.common import NUM
from tinyad.autoDiff.operators.binary_ops import Add, Mult, Exp
from tinyad.autoDiff.var import ConstantVar, Var
from tinyad.tests.operators.combined_ops_tests.combinedBaseTest import CombinedBaseTest

class BinaryOperatorsBaseTest(CombinedBaseTest):

    ########################### Multiplicative term helper functions ###########################

    def _create_multiplicative_term_int_expos(self, variables: List[Var], max_subset_size=5) -> Tuple[Var, Dict[int, int]]:
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


    def _calculate_multiplicative_term_value(self, variables: List[Var], exponents: Dict[int, int]) -> NUM:
        """Calculate the value of a single term (know to be the product of variables to some powers)."""
        term_value = 1.0
        for idx, power in exponents.items():
            term_value *= variables[idx].value ** power
        return term_value


    def _calculate_gradient_multiplicative_term(self, 
                                     variables: List[Var], 
                                     term_exponents: List[Dict[int, int]], 
                                     term_sign: NUM = 1) -> List[NUM]:
        """Calculate expected gradients for each variable from a list of terms."""
        if term_sign not in [-1, 0, 1]:
            raise ValueError("term_sign must be -1, 0, or 1")
        
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


    def _create_multiplicative_term_float_expos(self, variables: List[Var], max_subset_size=5) -> Tuple[Var, Dict[int, float]]:
        """
        Create a single polynomial term with random variables and powers using the Exp operator.
        This allows for floating-point exponents, unlike _create_multiplicative_term_int_expos.
        """
        n_vars = len(variables)
        subset_size = random.randint(1, min(max_subset_size, n_vars))
        var_indices = random.sample(range(n_vars), subset_size)
        
        # Assign random exponents (powers) to each variable in this term
        exponents = {}
        for idx in var_indices:
            # Use floating point exponents between 0.1 and 5.0
            exponents[idx] = round(random.uniform(0.1, 5.0), 2)
        
        # Create the product expression for this term
        term = ConstantVar("const", 1.0)
        for idx, power in exponents.items():
            # Use Exp operator instead of repeated multiplication
            var_term = Exp(variables[idx], ConstantVar(f"p_{idx}", power))
            term = Mult(term, var_term)
        
        # The exponents variable is a dictionary that maps the variable index (from the original `variables` list)
        # to the power to which the variable is raised in this term
        return term, exponents

    ########################### Additive term helper functions ###########################

    def _create_additive_term(self, 
                            variables: List[Var], 
                            max_subset_size=5) -> Tuple[Var, Dict[int, int]]:
        """Create a single polynomial term with random variables and powers."""
        # 2. Create two additive terms with random coefficients
        # For first term: select a subset of variables
        n_vars = len(variables)
        term_size = random.randint(1, min(max_subset_size, n_vars))
        term_indices = random.sample(range(n_vars), term_size)
        term_coeffs = {idx: round(random.uniform(-2, 2), 2) for idx in term_indices}
        

        # 3. Build the additive terms
        term = ConstantVar("zero", 0.0)
        for idx, coeff in term_coeffs.items():
            term = Add(term, Mult(ConstantVar(f"c_{idx}", coeff), variables[idx]))

        return term, term_coeffs


    def _calculate_additive_term_value(self, variables: List[Var], coefficients: Dict[int, NUM]) -> NUM:
        """Calculate the value of a single term (know to be the product of variables to some powers)."""
        return sum(coeff * variables[idx].value for idx, coeff in coefficients.items())


    def _calculate_gradient_multiplication_two_additive_terms(self, variables: List[Var], 
                                                            term1_coeffs: Dict[int, NUM], 
                                                            term2_coeffs: Dict[int, NUM]) -> List[NUM]:
        """
        Calculate expected gradients when two additive terms are multiplied.
        For expression (a₁x₁ + a₂x₂ + ... + aₙxₙ) * (b₁y₁ + b₂y₂ + ... + bₘyₘ),
        apply the product rule for differentiation.
        
        Args:
            variables: List of all variables
            term1_coeffs: Dictionary mapping variable index to its coefficient in first term
            term2_coeffs: Dictionary mapping variable index to its coefficient in second term
            
        Returns:
            List of expected gradients for each variable
        """
        n_vars = len(variables)
        gradients = [0] * n_vars
        
        # Calculate values of each additive term
        term1_value = sum(coeff * variables[idx].value for idx, coeff in term1_coeffs.items())
        term2_value = sum(coeff * variables[idx].value for idx, coeff in term2_coeffs.items())
        
        # For each variable, apply the product rule
        for idx in range(n_vars):
            # Check if the variable appears in either or both terms
            in_term1 = idx in term1_coeffs
            in_term2 = idx in term2_coeffs
            
            if in_term1 and in_term2:
                # Variable appears in both terms
                a_i = term1_coeffs[idx]
                b_i = term2_coeffs[idx]
                x_i = variables[idx].value
                
                # Calculate P_rest and Q_rest (the parts of the terms without x_i)
                term1_rest = term1_value - a_i * x_i
                term2_rest = term2_value - b_i * x_i
                
                # Apply the product rule for the specific case where variable appears in both terms
                gradient = 2 * a_i * b_i * x_i + a_i * term2_rest + b_i * term1_rest
            elif in_term1:
                # Variable appears only in term1
                gradient = term1_coeffs[idx] * term2_value
            elif in_term2:
                # Variable appears only in term2
                gradient = term2_coeffs[idx] * term1_value
            else:
                # Variable doesn't appear in either term
                gradient = 0
            
            gradients[idx] = gradient
        
        return gradients                    


    ########################### Additive-Exponential term helper functions ###########################
    def _calculate_gradient_multiplication_additive_term_and_multiplicative_term(self, 
                                                                                variables: List[Var], 
                                                                                additive_term_coeffs: Dict[int, NUM], 
                                                                                multiplicative_term_exponents: Dict[int, int]) -> List[NUM]:
        """Calculate expected gradients when an additive term is multiplied by a multiplicative term."""
        n_vars = len(variables)
        gradients = [0] * n_vars

        def calculate_derivative(coeff: NUM, power: NUM):
            if power == 0:
                return 0
            return coeff * power * (variables[idx].value ** (power - 1))

        for idx, coeff in additive_term_coeffs.items():
            for exp_idx, exp in multiplicative_term_exponents.items():
                gradients[idx] += calculate_derivative(coeff, exp) * variables[exp_idx].value

        return gradients


    def _create_additive_exponential_term_int_expos(self, variables: List[Var], max_power:NUM, max_subset_size=5) -> Tuple[Var, Dict[int, Tuple[NUM, int]]]:
        """
        Create a single additive-exponential term of the form: a₁x₁^p₁ + a₂x₂^p₂ + ... + aₙxₙ^pₙ
        
        Returns:
            Tuple of (term expression, coefficients_and_powers dict)
            where coefficients_and_powers maps variable index to (coefficient, power) tuple
        """
        n_vars = len(variables)
        term_size = random.randint(1, min(max_subset_size, n_vars))
        term_indices = random.sample(range(n_vars), term_size)
        
        # Dictionary mapping variable index -> (coefficient, power)
        coeffs_and_powers = {}
        for idx in term_indices:
            # Generate random coefficient and power for each variable
            coeff = round(random.uniform(-2, 2), 2)
            power = random.randint(1, max_power) 
            coeffs_and_powers[idx] = (coeff, power)
        
        # Build the additive-exponential term
        term = ConstantVar("zero", 0.0)
        for idx, (coeff, power) in coeffs_and_powers.items():
            # Create x^power
            var_term = variables[idx]
            for _ in range(power - 1):  # Already have one power from the variable itself
                var_term = Mult(var_term, variables[idx])
            
            # Add a_i * x_i^power to the sum
            term = Add(term, Mult(ConstantVar(f"c_{idx}", coeff), var_term))
        
        return term, coeffs_and_powers


    def _calculate_additive_exponential_term_value(self, variables: List[Var], 
                                                 coeffs_and_powers: Dict[int, Tuple[NUM, int]]) -> NUM:
        """Calculate the value of an additive-exponential term."""
        return sum(coeff * (variables[idx].value ** power) 
                  for idx, (coeff, power) in coeffs_and_powers.items())


    def _calculate_gradient_additive_exponential_term(self, variables: List[Var], 
                                                    coeffs_and_powers: Dict[int, Tuple[NUM, int]],
                                                    upstream_gradient: NUM = 1.0) -> List[NUM]:
        """
        Calculate gradient of an additive-exponential term with respect to each variable.
        For term of form: a₁x₁^p₁ + a₂x₂^p₂ + ... + aₙxₙ^pₙ
        The gradient for x_i is: a_i * p_i * x_i^(p_i-1) * upstream_gradient
        """
        n_vars = len(variables)
        gradients = [0] * n_vars
        
        for idx, (coeff, power) in coeffs_and_powers.items():
            if power > 0:  # Only variables with positive power have non-zero gradient
                gradients[idx] = coeff * power * (variables[idx].value ** (power - 1)) * upstream_gradient
        
        return gradients


    def _calculate_gradient_multiplication_two_additive_exponential_terms(self, 
                                                                        variables: List[Var],
                                                                        term1_coeffs_powers: Dict[int, Tuple[NUM, int]],
                                                                        term2_coeffs_powers: Dict[int, Tuple[NUM, int]]) -> List[NUM]:
        """
        Calculate expected gradients when two additive-exponential terms are multiplied.
        Applies the product rule: d/dx(f*g) = (df/dx)*g + f*(dg/dx)
        """
        n_vars = len(variables)
        gradients = [0] * n_vars
        
        # Calculate values of each additive-exponential term
        term1_value = self._calculate_additive_exponential_term_value(variables, term1_coeffs_powers)
        term2_value = self._calculate_additive_exponential_term_value(variables, term2_coeffs_powers)
        
        # For each variable, apply the product rule
        for idx in range(n_vars):
            gradient = 0
            
            # Check if variable appears in first term
            if idx in term1_coeffs_powers:
                coeff1, power1 = term1_coeffs_powers[idx]
                # df1/dx_i = coeff1 * power1 * x_i^(power1-1)
                if power1 > 0:
                    df1_dxi = coeff1 * power1 * (variables[idx].value ** (power1 - 1))
                    gradient += df1_dxi * term2_value
            
            # Check if variable appears in second term
            if idx in term2_coeffs_powers:
                coeff2, power2 = term2_coeffs_powers[idx]
                # df2/dx_i = coeff2 * power2 * x_i^(power2-1)
                if power2 > 0:
                    df2_dxi = coeff2 * power2 * (variables[idx].value ** (power2 - 1))
                    gradient += df2_dxi * term1_value
            
            gradients[idx] = gradient
        
        return gradients


    def _create_additive_exponential_term_float_expos(self, variables: List[Var], max_subset_size=5) -> Tuple[Var, Dict[int, Tuple[NUM, int]]]:
        """
        Create a single additive-exponential term of the form: a₁x₁^p₁ + a₂x₂^p₂ + ... + aₙxₙ^pₙ
        
        Returns:
            Tuple of (term expression, coefficients_and_powers dict)
            where coefficients_and_powers maps variable index to (coefficient, power) tuple
        """
        n_vars = len(variables)
        term_size = random.randint(1, min(max_subset_size, n_vars))
        term_indices = random.sample(range(n_vars), term_size)
        
        # Dictionary mapping variable index -> (coefficient, power)
        coeffs_and_powers = {}
        for idx in term_indices:
            # Generate random coefficient and power for each variable
            coeff = round(random.uniform(-2, 2), 2)
            power = round(random.uniform(0.1, 3.0), 2)
            coeffs_and_powers[idx] = (coeff, power)
        
        # Build the additive-exponential term
        term = ConstantVar("zero", 0.0)
        for idx, (coeff, power) in coeffs_and_powers.items():
            # Create x^power
            var_term = Exp(variables[idx], ConstantVar(f"p_{idx}", power))
            
            # Add a_i * x_i^power to the sum
            term = Add(term, Mult(ConstantVar(f"c_{idx}", coeff), var_term))
        
        return term, coeffs_and_powers


