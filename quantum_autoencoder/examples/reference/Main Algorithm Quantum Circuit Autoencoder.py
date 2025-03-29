import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit import (QuantumCircuit,
                            QuantumRegister, ClassicalRegister,
                            Parameter, ParameterVector)
from qiskit.quantum_info import state_fidelity, DensityMatrix, partial_trace
from qiskit.circuit.library import RealAmplitudes, U3Gate
from qiskit_algorithms.optimizers import L_BFGS_B, SLSQP, SPSA, COBYLA
from qiskit import Aer, execute
from qiskit.providers.fake_provider import FakeGuadalupeV2


def encoder(num_bits, reps, name):
    encoder = RealAmplitudes(num_bits, entanglement='full', reps=reps, name=name)
    return encoder


def encode_qc(num_qbits, num_trash, train_circuit, encoder_u, encoder_v, pars_u, pars_v):
    qr = QuantumRegister(num_qbits)
    qc_encode = QuantumCircuit(qr)
    u1 = encoder_u.assign_parameters(pars_u)
    v1 = encoder_v.assign_parameters(pars_v)
    qc_encode.compose(u1, qr, inplace=True)
    qc_encode.compose(train_circuit, qr, inplace=True)
    qc_encode.compose(v1, qr, inplace=True)
    return qc_encode


def swap_test(qc, cbit, trash_states, ref_states, swap_states, control_state):
    qc.h(control_state)
    qc.cswap(control_state, trash_states, swap_states[:1])
    qc.cswap(control_state, ref_states, swap_states[1:])
    qc.h(control_state)
    qc.measure(control_state, cbit)
    return qc


def loss_fun(theta):
    num_theta = len(theta)
    half_num_theta = int(num_theta/2)
    pars_u = theta[:half_num_theta]
    pars_v = theta[half_num_theta:]
    qc = QuantumCircuit(qr, trash_r, ref_r, aux_r, swap_r, test_r, cr)
    qc.h(aux_r)
    qc.cnot(aux_r, qr)
    qc.h(trash_r)
    qc.cnot(trash_r, ref_r)
    qc.h(swap_r[0])
    qc.cnot(swap_r[0], swap_r[1])
    loss_list = []
    for i_circuit in D_train:
        # 5: Apply encoders U(θit) and V(θit) on the training data Ei
        #    to get the channel Π = V(θit) ◦ E_i ◦ U(θit);
        # 6: Apply Π on a composite system A′ and C1 with states ωA′ and ϕ+ C1;
        pi_channel = encode_qc(num_qbits, num_trash, i_circuit, encoder_u, encoder_v, pars_u, pars_v)
        qc.compose(pi_channel, qubits=qr_bits, inplace=True)
        # 7: Apply a swap test on the “trash” system C2 and ancila system C1′ to estimate the
        #    fidelity between state of C2 and ϕ+ C1 and calculate Li 3(θit);
        qc = swap_test(qc, cr, trash_r, ref_r, swap_r, test_r)
        shots = 1024
        job = execute(qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        loss = counts['1']/shots
        loss_list.append(loss)
    total_loss = np.sum(loss_list)/len(D_train)
    opt_y.append(total_loss)
    return total_loss


train_circuit = QuantumCircuit(4)
train_circuit.h(0)
train_circuit.cnot(0, 1)
train_circuit.h(2)
train_circuit.cnot(2, 3)
D_train = [train_circuit]
#D_train = []

num_points = 10
num_qbits = 4
for i in range(1):
    circuit = QuantumCircuit(4)
    for i_qubit in range(num_qbits):
        circuit.u(np.random.random()*2*np.pi,
                  np.random.random()*2*np.pi,
                  np.random.random()*2*np.pi, i_qubit)
    D_train.append(circuit)
print('number of training circuit: ', len(D_train))


num_cbits = 1
num_trash = 1
qr = QuantumRegister(num_qbits - num_trash)
trash_r = QuantumRegister(num_trash, 'trash')
ref_r = QuantumRegister(num_trash, 'ref')
aux_r = QuantumRegister(num_qbits - num_trash, 'aux')
swap_r = QuantumRegister(num_trash*2, 'swap')
test_r = QuantumRegister(1, 'swap_test_control')
cr = ClassicalRegister(num_cbits)
num_circuit = len(D_train)
objective_func_vals = []
optimizer = SPSA(maxiter=200)
backend = Aer.get_backend('statevector_simulator')

qc = QuantumCircuit(qr, trash_r, ref_r, aux_r, swap_r, test_r, cr)
qr_bits = list(range(num_qbits))
trash_bits = list(range(num_qbits - num_trash, num_qbits))
ref_bits = list(range(num_qbits, num_qbits + num_trash))
aux_bits = list(range(num_qbits + num_trash, num_qbits + num_trash + num_qbits - num_trash))
swap_bits = list(range(num_qbits + num_trash + num_qbits - num_trash,
                       num_qbits + num_trash + num_qbits - num_trash + num_trash*2))
control_bit = [num_qbits + num_trash + num_qbits - num_trash + num_trash*2]

qc.h(aux_r)
qc.cnot(aux_r, qr)
qc.h(trash_r)
qc.cnot(trash_r, ref_r)
qc.h(swap_r[0])
qc.cnot(swap_r[0], swap_r[1])

encoder_u = encoder(num_qbits, reps=5, name='u1')
encoder_v = encoder(num_qbits, reps=5, name='v1')
num_u_theta = encoder_u.num_parameters
num_v_theta = encoder_v.num_parameters

# 1: Initialize θ randomly.
init_theta_u = list(np.random.rand(encoder_u.num_parameters))
init_theta_v = list(np.random.rand(encoder_v.num_parameters))
init_theta = []
init_theta.extend(init_theta_u)
init_theta.extend(init_theta_v)
pars_u = init_theta[:num_u_theta]
pars_v = init_theta[num_u_theta:]
converged = False
opt_y = []


# 2: while not converged or the iteration ITR is not satisfied do
n_iter = 0
while not converged or n_iter<200:
    # 3: Initialize loss L(θit) = 0;
    loss = 0
    # 4: for each quantum circuit E_i in D_train do

        # 8: Let L(θit)+ = n1Li 3(θit);
        #loss += loss_fun(theta)

        # 9: end for
    # 10: Update parameters θit+1 of L(·) using classical optimizer;
    opt_result = optimizer.minimize(fun=loss_fun, x0=init_theta)
    theta_opt = opt_result.x
    print(opt_result)
    print(opt_result.fun)
    plt.title(f"number of training circuit: {num_circuit}")
    plt.plot(range(len(opt_y)), opt_y)
    plt.show()
    # 11: end while
    n_iter += 1
    exit()
#return theta_opt
# 12: Output the optimal parameters θOP T ;



