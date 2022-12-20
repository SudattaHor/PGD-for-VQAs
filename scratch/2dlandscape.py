from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import Z, I
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn
import numpy as np
from qiskit.opflow import Gradient
from util.opt_log import OptimizerLog

# SIMULATOR
quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'),
                                   # we'll set a seed for reproducibility
                                   shots=8192, seed_simulator=2718,
                                   seed_transpiler=2718)
sampler = CircuitSampler(quantum_instance)


# ANSATZ
ansatz = RealAmplitudes(num_qubits=2, reps=1,
                        entanglement='linear').decompose()

# COST FUNCTION
hamiltonian = Z ^ Z
expectation = StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)  # todo
pauli_basis = PauliExpectation().convert(expectation)


def evaluate_expectation(theta):
    value_dict = dict(zip(ansatz.parameters, theta))
    result = sampler.convert(pauli_basis, params=value_dict).eval()
    return np.real(result)

# INITIAL POINT
# initial_point = np.random.random(ansatz.num_parameters)
initial_point = np.array([0.43253681, 0.09507794, 0.42805949, 0.34210341])

# OPTIMIZER
