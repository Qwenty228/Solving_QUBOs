from typing import Literal
from abc import ABC, abstractmethod


class BaseQUBO(ABC):
    def __init__(self):
        self.__solver = "Qiskit"
    @property
    def solver(self):
        return self.__solver
    
    @abstractmethod
    def qubo(self, solver: Literal['Amplify', 'Qiskit'], penalty: int = 10) -> dict:
        self.__solver = format

    @abstractmethod
    def interpret(self, result, solver: Literal['Qiskit', 'D-Wave', "Fixstar", "Gurobi"] = "Gurobi", verbose=False) -> tuple:
        self.__solver = solver

    @abstractmethod
    def brute_force(self) -> tuple:
        pass


