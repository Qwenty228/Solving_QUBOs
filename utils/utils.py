import random
import numpy as np

import re

from IPython.display import Math
from qiskit import QuantumCircuit

# auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]



def convert_indices_to_comma_notation(formula):
    # Regex to match x_{number1}_{number2}
    pattern = r"x_(\d+)\_(\d+)"
    # Replace with x_{n1, n2}
    result = re.sub(pattern, r"x_{\1, \2}", formula)
    return result

def formula_to_latex(formula):
    # Remove the "minimize" and "(16 variables..." parts
    clean_formula = formula.split("minimize", 1)[-1].split("(")[0].strip()
    
    # Replace mathematical symbols for LaTeX formatting
    latex_formula = clean_formula.replace("^2", "^{2}").replace("*", " ")
    latex_formula = convert_indices_to_comma_notation(latex_formula)
    # Add line breaks for better readability

    return Math(latex_formula)



def validate_initial_point(point: np.ndarray | None | None, circuit: QuantumCircuit) -> np.ndarray:
    r"""
    Validate a choice of initial point against a choice of circuit. If no point is provided, a
    random point will be generated within certain parameter bounds. It will first look to the
    circuit for these bounds. If the circuit does not specify bounds, bounds of :math:`-2\pi`,
    :math:`2\pi` will be used.

    Args:
        point: An initial point.
        circuit: A parameterized quantum circuit.

    Returns:
        A validated initial point.

    Raises:
        ValueError: If the dimension of the initial point does not match the number of circuit
        parameters.
    """
    expected_size = circuit.num_parameters

    if point is None:
        # get bounds if circuit has them set, otherwise use [-2pi, 2pi] for each parameter
        bounds = getattr(circuit, "parameter_bounds", None)
        if bounds is None:
            bounds = [(-2 * np.pi, 2 * np.pi)] * expected_size

        # replace all Nones by [-2pi, 2pi]
        lower_bounds = []
        upper_bounds = []
        for lower, upper in bounds:
            lower_bounds.append(lower if lower is not None else -2 * np.pi)
            upper_bounds.append(upper if upper is not None else 2 * np.pi)

        # sample from within bounds
        point = random.uniform(lower_bounds, upper_bounds)

    elif len(point) != expected_size:
        raise ValueError(
            f"The dimension of the initial point ({len(point)}) does not match the "
            f"number of parameters in the circuit ({expected_size})."
        )

    return point

def validate_bounds(circuit: QuantumCircuit) -> list[tuple[float | None, float | None]]:
    """
    Validate the bounds provided by a quantum circuit against its number of parameters.
    If no bounds are obtained, return ``None`` for all lower and upper bounds.

    Args:
        circuit: A parameterized quantum circuit.

    Returns:
        A list of tuples (lower_bound, upper_bound)).

    Raises:
        ValueError: If the number of bounds does not the match the number of circuit parameters.
    """
    if hasattr(circuit, "parameter_bounds") and circuit.parameter_bounds is not None:
        bounds = circuit.parameter_bounds
        if len(bounds) != circuit.num_parameters:
            raise ValueError(
                f"The number of bounds ({len(bounds)}) does not match the number of "
                f"parameters in the circuit ({circuit.num_parameters})."
            )
    else:
        bounds = [(None, None)] * circuit.num_parameters

    return bounds



# Example usage
if __name__ == "__main__":
    # formula = """minimize 30*x_0_0^2 + 30*x_0_0*x_0_1 + 30*x_0_0*x_0_2 + ... - 60*x_3_2 - 60*x_3_3 + 120 (16 variables, 0 constraints, '')"""
    # latex_result = formula_to_latex(formula)

    # # # Display in Jupyter Notebook
    # from IPython.display import display, Math
    # display(Math(latex_result))
    a, b = 1, 2
    x = [a , b ] * 10

    
    x[1] = 12
    print(x)