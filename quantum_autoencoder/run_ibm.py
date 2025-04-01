"""Script to run quantum autoencoder on IBM Quantum hardware."""

import numpy as np
from quantum_autoencoder.ibm_backend import get_ibm_backend, get_ibm_primitives
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import Session
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import SPSA
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def encoder(num_bits, reps, name):
    """Create an encoder circuit using RealAmplitudes."""
    return RealAmplitudes(num_bits, entanglement='full', reps=reps, name=name)

def encode_qc(num_qbits, num_trash, train_circuit, encoder_u, encoder_v, pars_u, pars_v):
    """Create the encoding quantum circuit."""
    qr = QuantumRegister(num_qbits)
    qc_encode = QuantumCircuit(qr)
    u1 = encoder_u.assign_parameters(pars_u)
    v1 = encoder_v.assign_parameters(pars_v)
    qc_encode.compose(u1, qr, inplace=True)
    qc_encode.compose(train_circuit, qr, inplace=True)
    qc_encode.compose(v1, qr, inplace=True)
    return qc_encode

def swap_test(qc, cbit, trash_states, ref_states, swap_states, control_state):
    """Implement the SWAP test for fidelity measurement."""
    qc.h(control_state)
    
    # Apply CSWAP gates for each pair of qubits
    for i in range(len(trash_states)):
        qc.cswap(control_state, trash_states[i], swap_states[i])
        qc.cswap(control_state, ref_states[i], swap_states[i + len(trash_states)])
    
    qc.h(control_state)
    qc.measure(control_state, cbit)
    return qc

def loss_fun(theta, qc, qr, trash_r, ref_r, aux_r, swap_r, test_r, cr, qr_bits, sampler, backend):
    """Calculate loss using SWAP test."""
    num_theta = len(theta)
    half_num_theta = int(num_theta/2)
    pars_u = theta[:half_num_theta]
    pars_v = theta[half_num_theta:]
    
    # Create circuit for this parameter set
    circuit = QuantumCircuit(qr, trash_r, ref_r, aux_r, swap_r, test_r, cr)
    
    # Initialize auxiliary states
    circuit.h(aux_r)
    for i in range(len(aux_r)):
        circuit.cx(aux_r[i], qr[i])
    
    # Initialize reference states
    circuit.h(trash_r)
    for i in range(len(trash_r)):
        circuit.cx(trash_r[i], ref_r[i])
    
    # Initialize swap register
    circuit.h(swap_r[0])
    circuit.cx(swap_r[0], swap_r[1])
    
    # Add encoding circuit
    pi_channel = encode_qc(num_qbits, num_trash, train_circuit, encoder_u, encoder_v, pars_u, pars_v)
    circuit.compose(pi_channel, qubits=qr_bits, inplace=True)
    
    # Add SWAP test
    circuit = swap_test(circuit, cr[0], trash_bits, ref_bits, swap_bits, control_bit)
    
    # Transpile circuit for the backend
    pm = generate_preset_pass_manager(
        optimization_level=3,
        target=backend.target
    )
    transpiled_circuit = pm.run(circuit)
    
    # Run circuit and get results
    job = sampler.run([transpiled_circuit])
    result = job.result()
    counts = result[0].data.meas.get_counts()
    
    # Calculate loss (probability of measuring |1⟩)
    loss = counts.get('1', 0) / sum(counts.values())
    opt_y.append(loss)
    
    return loss

def main():
    global num_qbits, num_trash, train_circuit, encoder_u, encoder_v
    global trash_bits, ref_bits, swap_bits, control_bit, opt_y
    
    # Circuit parameters (matching reference)
    num_qbits = 5  # Changed from 4 to match our setup
    num_trash = 2  # Changed from 1 to match our setup
    num_cbits = 1
    
    # Create quantum registers
    qr = QuantumRegister(num_qbits - num_trash)
    trash_r = QuantumRegister(num_trash, 'trash')
    ref_r = QuantumRegister(num_trash, 'ref')
    aux_r = QuantumRegister(num_qbits - num_trash, 'aux')
    swap_r = QuantumRegister(num_trash*2, 'swap')
    test_r = QuantumRegister(1, 'swap_test_control')
    cr = ClassicalRegister(num_cbits, 'meas')
    
    # Create training circuit (domain wall state |00111⟩)
    train_circuit = QuantumCircuit(num_qbits)
    for i in range(2, num_qbits):
        train_circuit.x(i)
    
    # Get IBM backend and service
    service, backend = get_ibm_backend("ibm_kyiv")
    
    # Create session and run within its context
    with Session(backend=backend) as session:
        sampler, _ = get_ibm_primitives(session, shots=8192)
        
        # Create encoders with same parameters as reference
        encoder_u = encoder(num_qbits, reps=5, name='u1')  # Changed to reps=5 to match reference
        encoder_v = encoder(num_qbits, reps=5, name='v1')  # Changed to reps=5 to match reference
        
        # Calculate number of parameters
        num_u_theta = encoder_u.num_parameters
        num_v_theta = encoder_v.num_parameters
        
        # Initialize parameters in [0,1] to match reference
        init_theta_u = list(np.random.rand(encoder_u.num_parameters))
        init_theta_v = list(np.random.rand(encoder_v.num_parameters))
        init_theta = []
        init_theta.extend(init_theta_u)
        init_theta.extend(init_theta_v)
        
        # Calculate qubit indices
        qr_bits = list(range(num_qbits))
        trash_bits = [num_qbits - num_trash + i for i in range(num_trash)]
        ref_bits = [num_qbits + i for i in range(num_trash)]
        aux_bits = [num_qbits + num_trash + i for i in range(num_qbits - num_trash)]
        swap_bits = [num_qbits + num_trash + num_qbits - num_trash + i for i in range(num_trash * 2)]
        control_bit = num_qbits + num_trash + num_qbits - num_trash + num_trash*2
        
        # Initialize optimization history
        opt_y = []
        
        # Create optimizer (matching reference)
        optimizer = SPSA(
            maxiter=200,
            learning_rate=0.15,
            perturbation=0.1,
            resamplings=1,
            trust_region=True
        )
        
        print("\nStarting optimization...")
        print(f"Number of parameters: {len(init_theta)}")
        print(f"Using {optimizer.__class__.__name__} optimizer with {optimizer.maxiter} iterations")
        
        # Run optimization
        opt_result = optimizer.minimize(
            fun=lambda x: loss_fun(x, QuantumCircuit(qr, trash_r, ref_r, aux_r, swap_r, test_r, cr),
                                 qr, trash_r, ref_r, aux_r, swap_r, test_r, cr, qr_bits, sampler, backend),
            x0=init_theta
        )
        
        print("\nOptimization complete!")
        print(f"Final loss: {opt_result.fun:.4f}")
        print(f"Final fidelity: {1 - 2*opt_result.fun:.4f}")
        
        # Save optimal parameters
        np.save("best_parameters_ibm_kyiv.npy", opt_result.x)
        print("\nOptimal parameters saved to best_parameters_ibm_kyiv.npy")

if __name__ == "__main__":
    main() 