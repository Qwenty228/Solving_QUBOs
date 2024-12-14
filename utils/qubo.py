from typing import Literal
from abc import ABC, abstractmethod


class BaseQUBO(ABC):
    def __init__(self):
        self.__solver = "qiskit"

    @abstractmethod
    def interpret(self, result, solver: Literal['Qiskit', 'D-Wave', "Fixstar", "Gurobi"] = "Gurobi", verbose=False) -> tuple:
        pass

    @abstractmethod
    def brute_force(self) -> tuple:
        pass


