import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from qiskit.circuit import (QuantumCircuit,
                            QuantumRegister, ClassicalRegister,
                            Parameter, ParameterVector)
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Choi
from qiskit_algorithms.optimizers import L_BFGS_B, SLSQP, SPSA, COBYLA
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.quantum_info import state_fidelity,DensityMatrix
from qiskit import Aer, execute


def swap_test(qc, ref_state, state1, state2):
    qc.h(ref_state)
    qc.cswap(control, state1, state2)
    qc.h(ref_state)
    return qc


def matrix_square_root(matrix):
    evalues, evectors = np.linalg.eig(a)
    sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
    return sqrt_matrix


def encoder(num_bits, reps, name):
    encoder = RealAmplitudes(num_bits, entanglement='full', reps=reps, name=name)
    return encoder


def identity_interpret(x):
    return x


def encode_qc(num_qbits, num_trash, train_circuit, encoder_u, encoder_v, pars_u, pars_v):
    qr = QuantumRegister(num_qbits)
    qc_encode = QuantumCircuit(qr)
    
    #encoder_u = RealAmplitudes(4, entanglement='full', reps=1, name='u1')
    #encoder_v = RealAmplitudes(4, entanglement='full', reps=1, name='v1')
    #pars_u = ParameterVector(name='u_theta', length=encoder_u.num_parameters)
    #pars_u = [0.0] * encoder_u.num_parameters
    #pars_v = ParameterVector(name='v_theta', length=encoder_v.num_parameters)
    #pars_v = [0.0] * encoder_v.num_parameters
#    pars_u = list(pars_u)
#    pars_v = list(pars_v)
    #encoder_u.decompose().draw('mpl')
    #plt.show()

    u1 = encoder_u.assign_parameters(pars_u)
    v1 = encoder_v.assign_parameters(pars_v)
    qc_encode.compose(u1, qr, inplace=True)
#    encode_qc.barrier()
    qc_encode.compose(train_circuit, qr, inplace=True)
#    encode_qc.barrier()
    qc_encode.compose(v1, qr, inplace=True)

    return qc_encode

def decode_qc(num_qbits, qc_encode, pars_u, pars_v):
    decode_qc = QuantumCircuit(num_qbits)
    #pars_u2 = ParameterVector(name='u2_theta', length=encoder_u.num_parameters)
    #pars_v2 = ParameterVector(name='v2_theta', length=encoder_v.num_parameters)
    pars_u2 = list(pars_u)
    pars_v2 = list(pars_v)
    u2 = encoder_u.assign_parameters(pars_u2)
    u2 = u2.inverse()
    v2 = encoder_v.assign_parameters(pars_v2)
    v2 = v2.inverse()
    decode_qc.compose(u2, inplace=True)
    decode_qc.compose(qc_encode.to_gate(), inplace=True)
    decode_qc.reset(3)
    decode_qc.compose(v2, inplace=True)
    return decode_qc

def fidelity(qc, train_circuit):
    choi_state = Choi(qc)
    epilson_choi = Choi(train_circuit)
    fidelity = state_fidelity(DensityMatrix.from_instruction(choi_state.to_instruction()),
                              DensityMatrix.from_instruction(epilson_choi.to_instruction()))
    return fidelity


def loss_1(qc, train_circuit, num_bits, num_circuit):
    mse = 0.0
    for i_epsilon in range(num_circuit):
        mse += fidelity(qc, train_circuit)**2
    loss = 1.0 - mse/num_bits
    return loss


def loss_fun(theta):
    # half parameter used in RealAmplitudes U, and half used in RealAmplitudes V
    # number_of_parameters = encoder.num_parameters
    pars_u = theta[:12]
    pars_v = theta[12:]
    #print(len(pars_u))
    #print(len(pars_v))
    
    #loss = loss_1(qc(num_qbits, num_trash, train_circuit, encoder_u, encoder_v, pars_u, pars_v), train_circuit, num_qbits, num_circuit)
    loss = loss_1(decode_qc(num_qbits,
                            encode_qc(num_qbits, num_trash, 
                                      train_circuit, encoder_u, encoder_v, 
                                      pars_u, pars_v),
                            pars_u, pars_v),
                  train_circuit, num_qbits, num_circuit)

    objective_func_vals.append(loss)
    return loss


num_qbits = 4
num_trash = 1
num_circuit = 1
objective_func_vals = []
encoder_u = encoder(num_qbits, reps=2, name='u1')
encoder_v = encoder(num_qbits, reps=2, name='v1')
init_theta_u = list(np.random.rand(encoder_u.num_parameters))
init_theta_v = list(np.random.rand(encoder_v.num_parameters))
#print(init_theta_u)
#print(init_theta_v)
init_theta = []
init_theta.extend(init_theta_u)
init_theta.extend(init_theta_v)

train_circuit = QuantumCircuit(4)
train_circuit.x(0)

qc_decode = decode_qc(num_qbits, encode_qc(num_qbits, num_trash, 
                                           train_circuit, encoder_u, encoder_v,
                                           init_theta_u, init_theta_v),
                      init_theta_u, init_theta_v)
qc_decode.draw('mpl')
plt.show()

#_qc = qc(num_qbits, num_trash, train_circuit, encoder_u, encoder_v, init_theta_u, init_theta_v)

optimizer = COBYLA(maxiter=100)

qnn = SamplerQNN(circuit=qc_decode,
                 input_params=init_theta,
                 weight_params=None,
                 interpret=identity_interpret,
                 output_shape=2,
                 )

opt_result = optimizer.minimize(fun=loss_fun, x0=init_theta)
print(init_theta)
#print(opt_result.x)
print(opt_result)
plt.plot(range(len(objective_func_vals)), objective_func_vals)
plt.show()
