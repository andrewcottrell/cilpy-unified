# cilpy/problem/cec2010.py
"""
This file wraps the 20 CEC 2010 large-scale functions from the `opfunu` library as
`Problem` objects so that cilpy solvers can be tested on them.
"""
import importlib.resources
import sys
import types
from typing import List, Tuple
import numpy as np
from cilpy.problem import Problem, Evaluation


def _load_cec2010():
    """Imports the CEC 2010 module of opfunu"""

    package_resource = types.ModuleType("pkg_resources")
    package_resource.resource_filename = (
        lambda package, resource: str(importlib.resources.files(package) / resource))
    
    sys.modules.setdefault("pkg_resources", package_resource)

    try:
        from opfunu.cec_based import cec2010

    except ImportError as error:
        raise ImportError("Install opfunu first") from error
    return cec2010

class CEC2010(Problem[List[float], float]):
    """A function from the CEC 2010 large-scale benchmark"""

    def __init__(self, number: int, dimension: int = 1000):
        """Initialises a CEC 2010 function.

        Args:
            number: The function number from 1 to 20
            dimension: The number of dimensions
        """
        if number < 1 or number > 20:
            raise ValueError("The function number must be 1 to 20.")

        cec2010 = _load_cec2010()
        function_class = getattr(cec2010, f"F{number}2010")
        self.function = function_class(ndim=dimension)

        lower_bounds = list(self.function.lb)
        upper_bounds = list(self.function.ub)
        super().__init__(dimension, (lower_bounds, upper_bounds), f"CEC2010-F{number}")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the function for a given solution."""

        fitness = self.function.evaluate(np.array(solution))

        return Evaluation(fitness=float(fitness))

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the functions are not dynamic."""

        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the functions are not multi-objective."""

        return False

if __name__ == "__main__":

    my_cec_function = CEC2010(2, 1000)
    optimum = list(my_cec_function.function.f_shift[:1000])
    print(my_cec_function.evaluate(optimum))
