import numpy as np
from matplotlib import pyplot as plt
from qiskit.primitives import StatevectorSampler as Sampler

from utils.utils import validate_initial_point

class QAOA:
    
    def __init__(self, circuit, reps = 2):
        initial_gamma = np.pi
        initial_beta = np.pi/2
        self.init_params = validate_initial_point([initial_gamma, initial_beta] * reps, circuit)
        self.objective_func_vals = [] 
        self.circuit = circuit


    def cost_func_estimator(self, params, hamiltonian, estimator):

        # transform the observable defined on virtual qubits to
        # an observable defined on all physical qubits
        isa_hamiltonian = hamiltonian.apply_layout(self.circuit.layout)

        pub = (self.circuit, isa_hamiltonian, params)
        job = estimator.run([pub])

        results = job.result()[0]
        cost = results.data.evs

        self.objective_func_vals.append(cost)
        
        return cost
    
    def draw_cost_function(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.objective_func_vals)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()
        return min(self.objective_func_vals)
    
