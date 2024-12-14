import rustworkx as rx
import numpy as np
import random
import matplotlib

from numpy.random import Generator, PCG64
from matplotlib import pyplot as plt
from utils import BaseQUBO, to_bitstring
from rustworkx.visualization import mpl_draw as draw_graph
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import SparsePauliOp

__all__ = ["TSP"]


class TSP(BaseQUBO):
    def __init__(self, n: int, seed: int=None, draw: bool=True):
        super().__init__()
        self.graph, self.weight_matrix = generate_tsp(n, seed, draw, 10)
        self.final_distribution_bin = None
        
    def qubo(self, solver, penalty = 10):
        super().qubo(solver, penalty)
        if solver == "Qiskit":
            qp = tsp_quad(self.weight_matrix)
            qp2qubo = QuadraticProgramToQubo(penalty=penalty)
            qubo = qp2qubo.convert(qp)
            qubitOp, offset = qubo.to_ising()
            return {"model": qubitOp, "offset":  offset}
        else:
            return {"model": None}
    
    def interpret(self, result, solver, verbose=False):
        super().interpret(result, solver, verbose)
        if solver == "Qiskit":
            counts_int = result[0].data.meas.get_int_counts()
            counts_bin = result[0].data.meas.get_counts()
            shots = sum(counts_int.values())
            final_distribution_int = {key: val/shots for key, val in counts_int.items()}
            self.final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
            if verbose:
                print(f"distribution: \n", final_distribution_int)
            
            n = len(self.graph)

            keys = list(final_distribution_int.keys())
            values = list(final_distribution_int.values())
            most_likely = keys[np.argmax(np.abs(values))]
            most_likely_bitstring = to_bitstring(most_likely, n**2)
            return reorder(most_likely_bitstring, n, reverse=True)
        else:
            return ()
    
    def plot_distribution(self, n: int=20):
        if self.final_distribution_bin and self.solver == "Qiskit":
            matplotlib.rcParams.update({"font.size": 10})

            final_bits_reduced = {key: value for key, value in sorted(self.final_distribution_bin.items(), key= lambda x: x[1], reverse=True)[:n]}
            for key, value in final_bits_reduced.items():
                print(f"bitstring: {reorder(key, len(self.graph), reverse=True)}, probability: {value}")

            fig = plt.figure(figsize=(11, 6))
            ax = fig.add_subplot(1, 1, 1)
            plt.xticks(rotation=45)
            plt.title("Result Distribution")
            plt.xlabel("Bitstrings")
            plt.ylabel("Probability")
            ax.bar(list(final_bits_reduced.keys()), list(final_bits_reduced.values()), color="tab:grey")

            plt.show()
        else:
            print("No distribution to plot")
    
    def brute_force(self):
        return super().brute_force()
    
    
def generate_tsp(n: int, seed: int, draw: bool, max_distance=100):
    """Generate a random graph for the Max-Cut problem.

    Args:
        n: number of nodes
        seed: seed for random number generator
        draw: whether to draw the graph
        max_distance: maximum distance between nodes
    
    Returns:
        graph: the generated graph
        weight_matrix: the weight matrix of the graph
    """
    if seed:
        rng = Generator(PCG64(seed))
    else:
        rng = np.random.default_rng()
    
    triangle = np.triu(rng.integers(
        1, max_distance, size=(n, n)), 1)  # Upper triangle of the matrix
    sym = triangle + triangle.T # Make the matrix symmetric
    # Set diagonal to zero, as distance to itself is zero
    np.fill_diagonal(sym, 0)
    graph = rx.PyGraph()
    graph.add_nodes_from(range(n)) 
    for i in range(n):
        for j in range(i+1, n):
            graph.add_edge(i, j, sym[i, j])
   
    if draw:
        draw_graph(graph, node_size=600, with_labels=True, edge_labels=str)
    return graph, sym


def tsp_quad(weights: np.ndarray) -> QuadraticProgram:
    """
    Create a Quadratic Program formulation for the Traveling Salesman Problem (TSP).

    The function takes a matrix of pairwise city distances and encodes the TSP 
    problem as a Quadratic Program (QP). The QP formulation aims to minimize 
    the total distance traveled by visiting each city exactly once and returning
    to the starting city.

    Parameters
    ----------
    weights : np.ndarray
        A 2D square matrix where `weights[i, j]` represents the distance between 
        city `i` and city `j`. The matrix must have shape `(n, n)`, where `n` is 
        the number of cities.

    Returns
    -------
    qp : QuadraticProgram
        A Quadratic Program representing the TSP problem. The objective is to minimize 
        the total distance of the tour, and the constraints ensure that each city is 
        visited exactly once and left exactly once.
"""
    qp = QuadraticProgram()

    size = weights.shape[0]

    cities = {}  # dictionary to store the city and order
    for i in range(size):
        for p in range(size):
            cities[f'x_{i}_{p}'] = qp.binary_var(f'x_{i}_{p}')

    # Objective function
    quadratic_matrix = {}
    for i in range(size):
        for j in range(size):
            if i != j:
                for p in range(size):
                    quadratic_matrix[(f'x_{i}_{p}', f'x_{j}_{(p+1) % size}')] = weights[i, j]

    qp.minimize(quadratic=quadratic_matrix)

    # Constraint 1: each city is visited exactly once
    for i in range(size):
        qp.linear_constraint(
            linear={f'x_{i}_{p}': 1 for p in range(size)}, sense='==', rhs=1)

    # Constraint 2: each city is left exactly once
    for p in range(size):
        qp.linear_constraint(
            linear={f'x_{i}_{p}': 1 for i in range(size)}, sense='==', rhs=1)

    return qp

def reorder(x: list, size: int, reverse=False):
    """
    Reorder the binary list to match the order of the cities.

    Given a binary list representing city visits in the TSP problem, this function 
    reorders the list to match the order in which cities are visited based on a 
    specific format.

    Parameters
    ----------
    x : list
        A binary list of length `size * size` representing the possible visitations
        of cities. A value of `1` at index `i` indicates that the city `(i % size)` 
        is visited at step `(i // size)`.
    size : int
        The number of cities involved in the TSP problem.

    Returns
    -------
    y : np.ndarray
        A reordered array of city indices, indicating the order in which the cities 
        are visited.

    Examples
    --------
    >>> x = [0, 1, 0, 1, 0, 1, 1, 0, 0]
    >>> size = 3
    >>> reorder(x, size)
    array([1., 2., 0.])
    """
    if reverse:
        x = x[::-1]
    y = np.zeros(size)
    for i, v in enumerate(x):
        if int(v) == 1:
            y[int(i) % size] = i // size
    return y
