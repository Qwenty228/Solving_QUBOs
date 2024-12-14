import rustworkx as rx
import numpy as np
import sympy
import re
import matplotlib

from matplotlib import pyplot as plt
from utils import BaseQUBO, to_bitstring
from rustworkx.visualization import mpl_draw as draw_graph
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import SparsePauliOp

__all__ = ["ThreeSat"]

class ThreeSat(BaseQUBO):
    def __init__(self, filename: str):
        super().__init__()
        if filename != None:
            self.clauses, self.num_n, self.num_m = read_cnf_file(filename)
        self.final_distribution_bin = None
        self.K = 0 # cost offset
        
    @classmethod
    def from_clauses(cls, clauses: np.ndarray, num_n: int):
        obj = cls(None)
        obj.clauses = clauses
        obj.num_n = num_n
        obj.num_m = clauses.shape[0]
        return obj
        
        
    def qubo(self, solver, penalty = 10):
        super().qubo(solver, penalty)
        if solver == "Qiskit":
            qp = sat_quad(self.clauses, self.num_n)
            qp2qubo = QuadraticProgramToQubo(penalty=penalty)
            qubo = qp2qubo.convert(qp)
            qubitOp, offset = qubo.to_ising()
            return {"model": qubitOp, "offset":  offset}
        else:
            return {}
        
        
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
            most_likely_bitstring = to_bitstring(most_likely, self.num_m + self.num_n)
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
    
    def verify(self, result) -> tuple[bool, int, list]:
        num_clauses_True = 0
        wrong_clause = []
        for clause in self.clauses:
            clause_Truth_value = False
            for literal in clause:
                if literal > 0 and result[literal - 1] == 1:
                    clause_Truth_value = True
                    break
                elif literal < 0 and result[abs(literal) - 1] == 0:
                    clause_Truth_value = True
                    break
            if clause_Truth_value:
                num_clauses_True += 1
            else:
                wrong_clause.append(clause)
        return (num_clauses_True == self.clauses.shape[0]), num_clauses_True, wrong_clause
    
def read_cnf_file(filename: str) -> tuple[np.ndarray, int, int]:
    clauses = []

    with open(filename, 'r') as file:
        for line in file:
            # Skip comment lines
            if line.startswith('c'):
                continue
            
            # Read the problem line to get the number of variables
            if line.startswith('p cnf'):
                parts = line.split()
                num_n = int(parts[2])
                num_m = int(parts[3])
                continue
            
            # Read clauses
            clause = [x for x in line.split() if x != "0"]  # Skip the trailing 0
            if (len(clause) == 3):
                clauses.append(clause)
    
    return np.array(clauses, dtype="int"), num_n, num_m





def sat_QUBO(clauses: np.ndarray[np.int_], num_literals: int) -> tuple[np.ndarray[np.int_], int]:
    num_x = num_literals 
    x = sympy.symbols(f'x0:{num_x}') # x0, x1, ... x_num_x -1
    num_m = clauses.shape[0] # clauses matrix Row
    w = sympy.symbols(f'w0:{num_m}') # w0, w1, ... w_num_m - 1
    QUBO_matrix = np.zeros((num_x + num_m, num_x + num_m), dtype=int)
    sum_g = 0

    error = False
    
    for i in range(num_m):
        x_array = clauses[i]
        # Make sure each clause is within specified literals.
        if (x_array[np.abs(x_array) > num_x].size > 0):
            error = True
            break

        y_i1 = x[x_array[0] - 1] if (x_array[0] > 0) else (1 + (-1 * x[x_array[0] * -1 - 1]))
        y_i2 = x[x_array[1] - 1] if (x_array[1] > 0) else (1 + (-1 * x[x_array[1] * -1 - 1]))
        y_i3 = x[x_array[2] - 1] if (x_array[2] > 0) else (1 + (-1 * x[x_array[2] * -1 - 1]))
        sum_g += y_i1 + y_i2 + y_i3 + (w[i] * y_i1) + (w[i] * y_i2) + (w[i] * y_i3) - (y_i1 * y_i2) - (y_i1 * y_i3) - (y_i2 * y_i3) - 2 * w[i]
    
    sum_neg_g = sympy.simplify(-1 * sum_g)
    sum_neg_g_dict = {str(term): coefficient for term, coefficient in sum_neg_g.as_coefficients_dict().items()}

    for term in sum_neg_g_dict:
        # print(f'{term}: {sum_neg_g_dict[term]}')
        if re.match(r'^w\d+$', term): # w[i]
            i = int(term[1:]) + num_x
            j = int(term[1:]) + num_x
            QUBO_matrix[i][j] = sum_neg_g_dict[term]
        elif re.match(r'^x\d+$', term): # x[i]
            i = int(term[1:])
            j = int(term[1:])
            QUBO_matrix[i][j] = sum_neg_g_dict[term]
        elif re.match(r'^w(0|[1-9][0-9]*)\*x(\d+)$', term): # w[i] * x[j]
            match = re.match(r'^w(0|[1-9][0-9]*)\*x(\d+)$', term)
            w_number = int(match.group(1))
            x_number = int(match.group(2))
            QUBO_matrix[x_number][w_number + num_x] = sum_neg_g_dict[term]
        elif re.match(r'^x(0|[1-9][0-9]*)\*x(\d+)$', term): # x[i] * x[j]
            match = re.match(r'^x(0|[1-9][0-9]*)\*x(\d+)$', term)
            first_x_number = int(match.group(1))
            second_x_number = int(match.group(2))
            QUBO_matrix[first_x_number][second_x_number] = sum_neg_g_dict[term]
        elif re.match(r'^1$', term):
            K = sum_neg_g_dict[term]
    
    if (error):
        return "An error has occured, Make sure the literals in each clause are within the specified literals."
    else:
        return QUBO_matrix, K
    
    
def sat_quad(clauses: np.ndarray[np.int_], num_literals: int) -> QuadraticProgram:
    qubo, _ = sat_QUBO(clauses, num_literals)
    qp = QuadraticProgram()
    for i in range(qubo.shape[0]):
        qp.binary_var(f'x{i}')

    qp.minimize(quadratic=qubo)
    return qp