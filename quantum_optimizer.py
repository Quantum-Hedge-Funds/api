# Import Qiskit packages
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.result import QuasiDistribution
import numpy as np

# The data providers of stock-market data
from qiskit_finance.applications.optimization import PortfolioDiversification

from qiskit_algorithms.utils import algorithm_globals

class QuantumOptimizer:
    def __init__(self, rho, n, q):
        self.rho = rho
        self.n = n
        self.q = q
        self.pdf = PortfolioDiversification(similarity_matrix=rho, num_assets=n, num_clusters=q)
        self.qp = self.pdf.to_quadratic_program()
        self.values = []

    # Obtains the least eigenvalue of the Hamiltonian classically
    def exact_solution(self):
        exact_mes = NumPyMinimumEigensolver()
        exact_eigensolver = MinimumEigenOptimizer(exact_mes)
        result = exact_eigensolver.solve(self.qp)
        return self.decode_result(result)

    def sampling_vqe_solution(self):
        def callback(evaluation_count, optimizer_parameters, estimated_value, metadata):
            self.values.append(np.real(estimated_value))
        algorithm_globals.random_seed = 100
        cobyla = COBYLA()
        cobyla.set_options(maxiter=250)
        ry = TwoLocal(self.n, "ry", "cz", reps=5, entanglement="full")
        svqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=cobyla, callback=callback)
        svqe = MinimumEigenOptimizer(svqe_mes)
        result = svqe.solve(self.qp)
        print("Result")
        print(result)
        return self.get_proportions(result)

    def circuit(self):
        ry = TwoLocal(self.n, "ry", "cz", reps=5, entanglement="full")
        return ry

    def qaoa_solution(self):
        algorithm_globals.random_seed = 1234
        cobyla = COBYLA()
        cobyla.set_options(maxiter=250)
        qaoa_mes = QAOA(sampler=Sampler(), optimizer=cobyla, reps=3)
        qaoa = MinimumEigenOptimizer(qaoa_mes)
        result = qaoa.solve(self.qp)
        return self.decode_result(result)

    def decode_result(self, result, offset=0):
        quantum_solution = 1 - (result.x)
        ground_level = self.qp.objective.evaluate(result.x)
        return quantum_solution, ground_level
    
    def get_proportions(self, result):
        selection = result.x
        value = result.fval
        print("Optimal: selection {}, value {:.4f}".format(selection, value))
        eigenstate = result.min_eigen_solver_result.eigenstate
        probabilities = (
            eigenstate.binary_probabilities()
            if isinstance(eigenstate, QuasiDistribution)
            else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
        )
        print("\n----------------- Full result ---------------------")
        print("selection\tvalue\t\tprobability")
        print("---------------------------------------------------")
        probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        proportions = [0 for _ in range(len(selection))]
        for k, v in probabilities:
            x = np.array([int(i) for i in list(reversed(k))])
            for j in range(len(x)):
                if x[j] == 1:
                    proportions[j] += v
            value = self.pdf.to_quadratic_program().objective.evaluate(x)
            print("%10s\t%.4f\t\t%.4f" % (x, value, v))
        return self.get_norm_proportions(proportions)

    def get_norm_proportions(self, proportions):
        print("Proportions", proportions)
        norm_proportions = []
        sum_proportions = sum(proportions[-self.n:])
        for y in proportions[-self.n:]:
            norm_proportions.append(y/sum_proportions)
        return norm_proportions
