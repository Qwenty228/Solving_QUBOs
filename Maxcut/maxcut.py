from utils import BaseQUBO
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
import random


class MaxCut(BaseQUBO):
    def __init__(self, n: int, seed: int=None, draw: bool=True):
        super().__init__()
        self.graph, self.weight_matrix = generate_maxcut(n, seed, draw)
        

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
    
    graph.add_edges_from(edges)
    if draw:
        draw_graph(graph, node_size=600, with_labels=True)
    return graph, matrix

