import rustworkx as rx
import numpy as np
import random

from utils import BaseQUBO
from rustworkx.visualization import mpl_draw as draw_graph
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import SparsePauliOp

__all__ = ["MaxCut"]

class MaxCut(BaseQUBO):
    def __init__(self, n: int, seed: int=None, draw: bool=True):
        super().__init__()
        self.graph, self.weight_matrix = generate_maxcut(n, seed, draw)
        
    def qubo(self, penalty = 10, format = "ising"):
        if format == "ising":
            self.__solver = "qiskit"
            qp = maxcut_quad(self.weight_matrix)
            qp2qubo = QuadraticProgramToQubo(penalty=penalty)
            qubo = qp2qubo.convert(qp)
            qubitOp, offset = qubo.to_ising()
            return {"model": qubitOp, "offset":  offset}
        else:
            return {}
        
    def build_max_cut_paulis(self) -> SparsePauliOp:
        """Convert the graph to Pauli list.

        This function does the inverse of `build_max_cut_graph`
        """
        pauli_list = []
        for edge in list(self.graph.edge_list()):
            paulis = ["I"] * len(self.graph)
            paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

            weight = self.graph.get_edge_data(edge[0], edge[1])

            pauli_list.append(("".join(paulis)[::-1], weight))

        return SparsePauliOp.from_list(pauli_list)


        
    def interpret(self, result, solver, verbose=False) -> tuple:
        pass

    def brute_force(self) -> tuple:
        pass
    
def generate_maxcut(n: int, seed: int=None, draw: bool=True) -> rx.PyGraph:
    """
    Generate a random graph for the MaxCut problem using rustworkx.
    
    Args:
        n (int): Number of nodes (minimum 4).
        seed (int): Seed for random number generator.
        draw (bool): Whether to draw the generated graph.
        
    Returns:
        graph (rx.PyGraph): Generated graph for MaxCut.
        matrix (np.ndarray): Adjacency matrix of the generated graph.
    """
    if n < 4:
        raise ValueError("Number of nodes 'n' must be at least 3.")
    
    graph = rx.PyGraph()
    graph.add_nodes_from(range(n)) 
    
    if seed is not None:
        random.seed(seed)
    
    matrix = np.matrix(np.zeros((n, n)))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if random.choice([True, False]):  
                edges.append((i, j, 1.0))
                matrix[i, j] = 1
                matrix[j, i] = 1
    
    graph.add_edges_from(edges)
    if draw:
        draw_graph(graph, node_size=600, with_labels=True)
    return graph, matrix


def maxcut_quad(weight_matrix: np.ndarray) -> QuadraticProgram:
    size = weight_matrix.shape[0]
    # convert maxcut cost function into a Quadratic Program
    qubo_matrix = np.zeros([size, size]) # Q
    qubo_linear = np.zeros(size) # c

    for i in range(size):
        for j in range(size):
            qubo_matrix[i, j] += weight_matrix[i, j]
            qubo_linear[i] -= weight_matrix[i, j]

  
    # Create an instance of a Quadratic Program
    qp = QuadraticProgram("Maxcut")
    for i in range(size):
        qp.binary_var(f"x_{i}")
    qp.minimize(quadratic=qubo_matrix, linear=qubo_linear)

    return qp


