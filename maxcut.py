import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np


def create_graph(n=5):
    graph = rx.PyGraph()
    graph.add_nodes_from(np.arange(0, n, 1))
    edge_list = [(0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
    graph.add_edges_from(edge_list)
    draw_graph(graph, node_size=600, with_labels=True)
    return graph
    

def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = graph.get_edge_data(edge[0], edge[1])

        pauli_list.append(("".join(paulis)[::-1], weight))

    return pauli_list


