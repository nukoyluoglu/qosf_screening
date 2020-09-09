from pyquil import Program, get_qc
from pyquil.gates import RZ, RX, CZ, MEASURE
from pyquil.latex import to_latex
from pyquil.api import local_forest_runtime
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt

# maps measurements 0 -> [1, 0], 1 -> [0, 1]
def state_to_vector(state):
        return [1, 0] if state == 0 else [0, 1]

# maps measurement of N qubits to 2^N state vector
def measurement_to_state_vector(measurement):
    state_vector = state_to_vector(measurement[0])
    for q in range(1, len(measurement)):
        state_vector = np.kron(state_vector, state_to_vector(measurement[q]))
    return state_vector     

with local_forest_runtime():      
    qc = get_qc('9q-square-qvm')
    qubits = [0, 1, 2, 3]
    dist_vs_L = {}

    # loops over numbers of layers L
    for L in range(1, 4):
        p = Program()
        ro = p.declare('ro', 'BIT', len(qubits))
        theta = p.declare('theta', 'REAL', 2 * L * len(qubits))

        # adds circuit blocks
        for block in range(2 * L):
            # even blocks
            if block % 2 == 0:
                for q in qubits:
                    theta_index = block * len(qubits) + q
                    p += RZ(theta[theta_index], q)
                for q1, q2 in combinations(qubits, 2):
                    p += CZ(q1, q2)
            # odd blocks
            else:
                for q in qubits:
                    theta_index = block * len(qubits) + q
                    p += RX(theta[theta_index], q)

        # adds measurements for all qubits, sets 1000 shots of program
        for q in qubits:
            p += MEASURE(q, ro[q])
        p.wrap_in_numshots_loop(1000)
        executable = qc.compile(p)
        
        def optimization_fn(theta):
            measurements = qc.run(executable, {'theta': theta})
            # computes wavefunction of circuit output state by averaging over 1000 samples
            psi_theta = np.mean([measurement_to_state_vector(measurement) for measurement in measurements], axis=0)
            # computes random 4-qubit state vector
            psi_rand = np.random.uniform(size=2 ** len(qubits))
            # returns sum of squares of distance vector components
            return np.vdot(psi_theta - psi_rand, psi_theta - psi_rand)
        
        # initializes random angles and sets angle domains [0, 2 * pi]
        theta_init = np.zeros(2 * L * len(qubits))
        theta_bounds = [(0, 2 * np.pi) for theta in theta_init]
        # optimizes angles to minimize distance between circuit output state and random state
        classical_optimization = minimize(optimization_fn, theta_init, bounds=theta_bounds)
        if classical_optimization.get('success'):
            theta_optimal = classical_optimization.get('x')
            dist_vs_L[L] = optimization_fn(theta_optimal)
        else:
            raise RuntimeError(classical_optimization.get('message'))

    # plotting
    fig = plt.figure()
    plt.plot(list(dist_vs_L.keys()), list(dist_vs_L.values()))
    plt.xlabel('Number of Layers (L)')
    plt.xticks(list(dist_vs_L.keys()))
    plt.ylabel('Minimum Distance from Random 4-Qubit State')
    plt.title('Distance of Task 1 Circuit 4-Qubit State from Random 4-Qubit State')
    plt.savefig('task_1_output.png')