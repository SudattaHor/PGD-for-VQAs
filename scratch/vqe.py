from qiskit import Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

backend = Aer.get_backend("aer_simulator")

def ansatz(params):
    qr = QuantumRegister(1, name="q")
    cr = ClassicalRegister(1, name='c')
    qc = QuantumCircuit(qr, cr)
    qc.u(params[0], params[1], params[2], qr[0])
    qc.measure(qr, cr[0])
    return qc


