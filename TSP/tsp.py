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
            pass
    
    def interpret(self, result, solver = "Gurobi", verbose=False):
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



