"""Test script for IBM Quantum backend configuration."""

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session

def main():
    """Run a simple test circuit on IBM Quantum hardware."""
    # Initialize service and get backend
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.least_busy(operational=True, simulator=False)
    
    # Create a simple test circuit
    qc = QuantumCircuit(2)
    qc.h(0)  # Hadamard on first qubit
    qc.cx(0, 1)  # CNOT with first qubit as control, second as target
    qc.measure_all()  # This creates a classical register named 'meas'
    
    print("\nTest circuit:")
    print(qc)
    
    # Transpile circuit for the backend
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)
    
    # Create session and run within its context
    with Session(backend=backend) as session:
        # Import primitives here to ensure they use the session context
        from quantum_autoencoder.ibm_backend import get_ibm_primitives
        
        # Get primitives configured for the session
        sampler, estimator = get_ibm_primitives(
            session=session,
            shots=1024
        )
        
        # Run the circuit
        print("\nExecuting circuit...")
        job = sampler.run([isa_circuit])  # V2 format: list of circuits
        result = job.result()
        
        # Get counts from the classical register "meas"
        counts = result[0].data.meas.get_counts()
        print("\nResults:")
        print(f"Counts: {counts}")

if __name__ == "__main__":
    main() 