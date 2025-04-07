
import random
import unittest
from typing import Callable, List, Optional

from tinyad.autoDiff.common import NUM
from tinyad.autoDiff.var import ConstantVar, ElementaryVar, Var


class CombinedBaseTest(unittest.TestCase):
    ########################### General helper functions ###########################

    def _create_variables(self, n_vars):
        """Helper to create a list of ElementaryVar objects with random values."""
        return [ElementaryVar(f"x{i}", round(random.uniform(0.5, 2.0), 3)) for i in range(n_vars)]



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
    


    def _verify_gradients(self, variables: List[Var], expected_gradients: List[NUM], counts: List[NUM], places: Optional[int]=None):
        """Verify that the actual gradients match expected gradients."""
        for i, var in enumerate(variables):
            if counts[i] == 0:
                self.assertTrue(
                    var.grad is None or var.grad == 0,
                    f"Variable {i} wasn't used but has gradient {var.grad}"
                )

            else:
                self.assertAlmostEqual(
                    var.grad, 
                    expected_gradients[i],
                    places=places,
                    msg=f"Variable {i} gradient incorrect. Expected {expected_gradients[i]}, got {var.grad}"
                )
