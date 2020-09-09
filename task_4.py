from pyquil import Program, get_qc
from pyquil.gates import RX, RY, CNOT, H, MEASURE
from pyquil.api import local_forest_runtime
import numpy as np
import matplotlib.pyplot as plt

# The matrix decomposed into a sum of terms Pauli terms is II + ZZ - 1/2 (XX + YY)
# Ansatz: (RX I) CX (H I) |00>
# Rotations for measurement basis:
## 0. II: constant, no measurements
## 1. ZZ: no rotations
## 2. XX: RY(-pi/2) rotation for each qubit
## 3. YY: RX(pi/2) rotation for each qubit

# prepares ansatz (RX I) CX (H I) |00>
def prepare_ansatz():
    ansatz = Program(H(0), CNOT(0, 1))
    ro = ansatz.declare('ro', 'BIT', 2)
    theta = ansatz.declare('theta', 'REAL')
    ansatz += RX(theta, 0)
    return ansatz, ro, theta

# adds measurements for qubits 0 and 1, sets 1000 shots of program
def prepare_measurement(p, ro):
    p += MEASURE(0, ro[0])
    p += MEASURE(1, ro[1])
    p.wrap_in_numshots_loop(1000)
    return qc.compile(p)

# maps measurements 0 -> 1, 1 -> -1
def get_eigenvalue(q):
    return - (2 * q - 1)

# measures qubits 0 and 1, computes expectation value for given executable corresponding to a Pauli term
def get_pauli_term(executable, theta):
    measurements = qc.run(executable, {'theta': [theta]})
    pauli_terms = [get_eigenvalue(measurement[0]) * get_eigenvalue(measurement[1]) for measurement in measurements]
    return np.mean(pauli_terms, axis=0)

with local_forest_runtime():
    qc = get_qc('9q-square-qvm')

    # Pauli term: ZZ, no rotations
    ansatz_1, ro_1, theta_1 = prepare_ansatz()
    executable_ZZ = prepare_measurement(ansatz_1, ro_1)

    # Pauli term: XX, RY(-pi/2) rotation for each qubit
    ansatz_2, ro_2, theta_2 = prepare_ansatz()
    ansatz_2 += RY(- np.pi / 2, 0)
    ansatz_2 += RY(- np.pi / 2, 1)
    executable_XX = prepare_measurement(ansatz_2, ro_2)

    # Pauli term: YY, RX(pi/2) rotation for each qubit
    ansatz_3, ro_3, theta_3 = prepare_ansatz()
    ansatz_3 += RX(np.pi / 2, 0)
    ansatz_3 += RX(np.pi / 2, 1)
    executable_YY = prepare_measurement(ansatz_3, ro_3)

    expectation_vs_theta = {}
    # grid search over theta in [0, 2 * pi]
    for theta in np.linspace(0, 2 * np.pi, 200):
        # computes expectation values of each Pauli term (without coefficients)
        pauli_term_ZZ = get_pauli_term(executable_ZZ, theta)
        pauli_term_XX = get_pauli_term(executable_XX, theta)
        pauli_term_YY = get_pauli_term(executable_YY, theta)
        # computes expectation value of Hamiltonian II + ZZ - 1/2 (XX + YY)
        expectation = 1 + pauli_term_ZZ - 0.5 * (pauli_term_XX + pauli_term_YY)
        expectation_vs_theta[theta] = expectation

    theta_optimal = min(expectation_vs_theta, key=expectation_vs_theta.get)
    min_expectation = expectation_vs_theta[theta_optimal]

    # plotting
    fig = plt.figure()
    plt.plot(list(expectation_vs_theta.keys()), list(expectation_vs_theta.values()),)
    plt.axvline(theta_optimal, linestyle='dashed', color='g', label='optimal angle = {}\nlowest eigenvalue = {}'.format(theta_optimal, min_expectation))
    plt.xlabel('Angle')
    plt.ylabel('Expectation Value of Hamiltonian')
    plt.title('VQE on Task 4 Hamiltonian (1000 shots per angle)')
    plt.legend()
    plt.savefig('task_4_output.png')