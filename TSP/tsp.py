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
        return super().interpret(result, solver, verbose)
    
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