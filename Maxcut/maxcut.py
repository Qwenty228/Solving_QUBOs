import rustworkx as rx
import numpy as np
import random
import matplotlib

from matplotlib import pyplot as plt
from utils import BaseQUBO, to_bitstring
from rustworkx.visualization import mpl_draw as draw_graph
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import SparsePauliOp

__all__ = ["MaxCut"]

class MaxCut(BaseQUBO):
    def __init__(self, n: int, seed: int=None, draw: bool=True):
        super().__init__()
        self.graph, self.weight_matrix = generate_maxcut(n, seed, draw)
        self.final_distribution_bin = None
        
    def qubo(self, solver, penalty = 10):
        super().qubo(solver, penalty)
        if solver == "Qiskit":
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
        super().interpret(result, solver, verbose)
        if solver == "Qiskit":
            counts_int = result[0].data.meas.get_int_counts()
            counts_bin = result[0].data.meas.get_counts()
            shots = sum(counts_int.values())
            final_distribution_int = {key: val/shots for key, val in counts_int.items()}
            self.final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
            if verbose:
                print(f"distribution: \n", final_distribution_int)
            

            keys = list(final_distribution_int.keys())
            values = list(final_distribution_int.values())
            most_likely = keys[np.argmax(np.abs(values))]
            most_likely_bitstring = to_bitstring(most_likely, len(self.graph))
            most_likely_bitstring.reverse()
            return most_likely_bitstring
        else:
            return ()
        
    def plot_distribution(self, n: int=20):
        if self.solver == 'Qiskit' and self.final_distribution_bin is not None:
            matplotlib.rcParams.update({"font.size": 10})

            final_bits_reduced = {str(key): value for key, value in sorted(self.final_distribution_bin.items(), key= lambda x: x[1], reverse=True)[:n]}

            fig = plt.figure(figsize=(11, 6))
            ax = fig.add_subplot(1, 1, 1)
            plt.xticks(rotation=45)
            plt.title("Result Distribution")
            plt.xlabel("Bitstrings (reversed)")
            plt.ylabel("Probability")
            ax.bar(list(final_bits_reduced.keys()), list(final_bits_reduced.values()), color="tab:grey")

            plt.show()
                        
        else:
            print("No distribution to draw.")
            
    def draw_result(self, result):
        # auxiliary function to plot graphs
        colors = ["tab:grey" if i == 0 else "tab:purple" for i in result]
        pos, default_axes = rx.spring_layout(self.graph), plt.axes(frameon=True)
        rx.visualization.mpl_draw(self.graph, node_color=colors, node_size=100, alpha=0.8, pos=pos)

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


