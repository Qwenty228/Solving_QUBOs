import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import numpy as np
import json

from amplify import VariableGenerator, ConstraintList, one_hot, Poly
from typing import Literal
from itertools import permutations
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

class TSP:
    """
    traveling salesman problem (TSP) class
    """
    def __init__(self, num_cities: int = 4, initial_plot: bool = False):
        self.__solver = "qiskit"
        self.n = num_cities
        self.distance_matrix = randomize_cities(num_cities, plot=initial_plot)
    
    @classmethod
    def from_distance_matrix(cls, distance_matrix: np.ndarray| list) -> 'TSP':
        tsp = cls()
        tsp.distance_matrix = np.array(distance_matrix)
        tsp.n = tsp.distance_matrix.shape[0]
        return tsp
    
    def qubo(self, penalty: int = 10, format: Literal['amplify', 'ising'] = "ising"):
        if format == 'amplify':
            self.__solver = "amplify"
            return create_tsp_qp_model(self.distance_matrix, weight=penalty)
        else:
             # converts the QUBO to an Ising Hamiltonian
             # making quadratic program from TSP weights
            self.__solver = "qiskit"
            qp = make_tsp_qp(self.distance_matrix)
            qp2qubo = QuadraticProgramToQubo(penalty=penalty)
            qubo = qp2qubo.convert(qp)
            qubitOp, offset = qubo.to_ising()
            return {"model": qubitOp, "offset":  offset}

    def interpret(self, result, solver: Literal['Qiskit', 'D-Wave', "Fixstar", "Gurobi"] = "Gurobi", verbose = False) -> tuple:
        if solver:
            assert solver in ['Qiskit', 'D-Wave', "Fixstar", "Gurobi"], f"Invalid solver, {solver} not in ['Qiskit', 'D-Wave', 'Fixstar', 'Gurobi']"
            self.__solver = solver
        
        if self.__solver == "Qiskit":
            reordered = reorder(result, self.n)
            dur = 1
            cost = 0
        else:
            reordered = reorder(list(result.best.values.values()), self.n)
            dur = result.execution_time.total_seconds()
            cost = result.best.objective

        if verbose:
            print(f"Cost: {cost}")
            print(f"Optimal tour: {reordered}")
            print(f"Execution time: {dur}")
        return dur, cost, reordered

    def brute_force(self)-> tuple:
        """
        Brute force solution for the Traveling Salesman Problem (TSP).

        This function evaluates all possible routes between cities and selects the one 
        with the shortest total distance using a brute force method. The brute force method
        checks every possible permutation of city visits.

        Parameters
        ----------
        weights : np.ndarray
            A 2D matrix where `weights[i, j]` represents the distance between city `i` 
            and city `j`. The matrix should have shape `(n, n)`.

        Returns
        -------
        min_path : tuple
            The path (as a tuple of city indices) that results in the shortest tour.
        min_cost : float
            The total distance of the optimal tour.

        Examples
        --------
        >>> weights = np.array([[0, 10, 15, 20],
        ...                     [10, 0, 35, 25],
        ...                     [15, 35, 0, 30],
        ...                     [20, 25, 30, 0]])
        >>> min_path, min_cost = brute_force_tsp(weights)
        >>> print(min_path, min_cost)
        (0, 1, 3, 2) 80

        Notes
        -----
        This method may be computationally expensive for large numbers of cities 
        due to the factorial growth of possible routes.
        """

        size = self.distance_matrix.shape[0]

        min_cost = np.inf
        min_path = None

        for path in permutations(range(size)):
            cost = 0
            for i in range(size):
                cost += self.distance_matrix[path[i], path[(i+1) % size]]

            if cost < min_cost:
                min_cost = cost
                min_path = path

        return min_path, min_cost   
    
    
    def draw_cities(self) -> None:
        """
        Draw the graph representing the TSP problem.

        This function creates a graph where the nodes represent cities and the edges 
        represent the pairwise distances between them. The graph is visualized using 
        the `networkx` library, with the distances shown on the edges.

        Returns
        -------
        None
            This function doesn't return anything. It directly visualizes the graph.

        Examples
        --------
        >>> tsp = TSP(4)
        >>> tsp.draw_cities()
        """
        n = self.distance_matrix.shape[0]

        G = nx.Graph()
        G.add_nodes_from(np.arange(n))
        colors = ['r' for node in G.nodes()]
        pos = nx.spring_layout(G)

        for i in range(n):
            for j in range(i):
                G.add_edge(i, j, weight=self.distance_matrix[i, j])

        default_axes = plt.axes(frameon=True)
        nx.draw_networkx(G, node_color=colors, node_size=600,
                        alpha=0.8, ax=default_axes, pos=pos)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
   
    def draw_tour(self, path: list) -> None:
        """
        Draw the optimal tour for the TSP problem.

        This function visualizes the optimal tour using the `networkx` library. The tour 
        is shown as a directed graph, where each edge corresponds to a leg of the tour.

        Parameters
        ----------
        path : list
            A list of city indices representing the optimal tour (i.e., the sequence in 
            which the cities are visited).

        Returns
        -------
        None
            This function doesn't return anything. It directly visualizes the tour.

        Examples
        --------
        >>> tsp = TSP(4)
        >>> path = [0, 1, 3, 2]
        >>> tps.draw_tour(path)
        """
        
        G = nx.DiGraph()
        G.add_nodes_from(np.arange(self.distance_matrix.shape[0]))
        pos = nx.spring_layout(G)
        colors = ['r' for node in G.nodes()]
        n = len(path)
        for i in range(n):
            j = (i + 1) % n
            G.add_edge(path[i], path[j], weight=self.distance_matrix[path[i], path[j]])
        default_axes = plt.axes(frameon=True)
        nx.draw_networkx(
            G, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
        )
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(
            G, pos, font_color="b", edge_labels=edge_labels)

 
def randomize_cities(num_nodes: int, plot: bool=True) -> np.ndarray:
    """
    Randomize the cities and return the pairwise distance matrix.

    This function generates random positions for a specified number of cities 
    on a 2D plane and calculates the Euclidean distance between each pair of cities.
    The distances are returned in a 2D matrix.

    Parameters
    ----------
    num_nodes : int
        The number of cities to randomize and include in the problem.
    plot : bool
        A flag to indicate whether to plot the cities on a 2D plane. The default is `True`.

    Returns
    -------
    dist : np.ndarray
        A 2D matrix of shape `(num_nodes, num_nodes)` where each element represents
        the distance between two cities. The values are scaled and rounded to integers.

    Examples
    --------
    >>> num_nodes = 5
    >>> dist = randomize_cities(num_nodes)
    >>> print(dist)
    array([[ 0, 10,  8, 11, 14],
           [10,  0,  9, 12,  7],
           [ 8,  9,  0, 10, 15],
           [11, 12, 10,  0, 13],
           [14,  7, 15, 13,  0]])
    """

    positions = np.random.rand(num_nodes, 2)
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(positions[:, 0], positions[:, 1], color='blue')

        for i, (x, y) in enumerate(positions):
            ax.text(x, y, f'N{i}', fontsize=12, ha='right')

        plt.title("Randomized cities")
        plt.grid(True)
        plt.show()

    # pairwise distance using Euclidean distance
    dist = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist[i, j] = np.linalg.norm(positions[i] - positions[j])

    return (dist * 10).astype(int)


def reorder(x: list, size: int):
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
    y = np.zeros(size)
    for i, v in enumerate(x):
        if v == 1:
            y[int(i) % size] = i // size
    return y


def make_tsp_qp(weights: np.ndarray) -> QuadraticProgram:
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


def create_tsp_qp_model(distance_matrix: np.ndarray, weight=1):
    size = distance_matrix.shape[0]

    gen = VariableGenerator()
    # Facility i on Site j
    q = gen.array(type="Binary", shape=(size, size), name="q")

    obj = Poly()

    for i in range(size):
        for j in range(size):
            if i != j:
                for p in range(size):
                    obj += q[i, p] * q[j, (p+1) % size] * distance_matrix[i, j]
              
    c = ConstraintList()

    for i in range(size):
        c += one_hot(q[i])
        c += one_hot(q[:, i].sum())

    c *= weight

    model = obj + c

    return {"model": model, "obj": obj, "c": c, "q": q}

def dense_distance_matrix(n, max_distance=100):
    """Create a dense distance matrix where all nodes are connected."""
    triangle = np.triu(np.random.randint(
        1, max_distance, size=(n, n)), 1)  # Upper triangle of the matrix
    sym = triangle + triangle.T  # Make the matrix symmetric
    # Set diagonal to zero, as distance to itself is zero
    np.fill_diagonal(sym, 0)
    return sym.tolist()



def __create_dst_mtx():
    with open("problems.json", mode="w") as f:
        problems = {
            "dense": {
                "low": {},
                "high": {}
            },
            "sparse": {
                "low": {},
                "high": {}
            }
        }
        json.dump(problems, f, indent=2)
    return problems


def get_distance_matrices(nodes: list[int] = [4, 5, 6, 7, 8, 10, 11, 12, 15, 20, 69]):
    if not os.path.exists("problems.json"):
        problems = __create_dst_mtx()

    with open("problems.json") as f:
        if os.stat("problems.json").st_size == 0:
            problems = __create_dst_mtx()
        else:
            problems = json.load(f)

    dense = problems["dense"]
    changes = False
    for i in nodes:
        if str(i) not in dense["low"]:
            dense["low"][str(i)] = dense_distance_matrix(i, max_distance=10)
            changes = True
        if str(i) not in dense["high"]:
            dense['high'][str(i)] = dense_distance_matrix(i, max_distance=1000)
            changes = True
            
    if changes:
        with open("problems.json", mode="w") as f:
            json.dump(problems, f, indent=2)

    return problems