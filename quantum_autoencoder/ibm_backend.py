"""IBM Quantum backend integration for quantum autoencoder."""

import os
from dotenv import load_dotenv
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Session,
    SamplerV2 as Sampler,
    EstimatorV2 as Estimator
)

def get_service():
    """Get the IBM Quantum service instance."""
    # Load API token from environment
    load_dotenv()
    token = os.getenv('IBM_QUANTUM_TOKEN')
    
    if not token:
        raise ValueError("IBM Quantum API token not found. Please set IBM_QUANTUM_TOKEN in .env file")
    
    # Initialize the IBM Quantum service
    return QiskitRuntimeService(channel="ibm_quantum", token=token)

def list_backends():
    """List all available IBM Quantum backends."""
    service = get_service()
    
    # List all available backends
    print("\nAvailable backends:")
    for backend in service.backends():
        status = backend.status()
        print(f"- {backend.name}: {'🟢 Ready' if status.operational else '🔴 Not operational'}")
    
    return service

def get_ibm_backend(backend_name: str = "ibm_brisbane"):
    """
    Get an IBM Quantum backend for running circuits.
    
    Args:
        backend_name: Name of the IBM Quantum backend to use
        
    Returns:
        The configured backend instance
    """
    service = get_service()
    
    # Get the backend
    backend = service.backend(backend_name)
    status = backend.status()
    
    print(f"\nSelected backend: {backend.name}")
    print(f"Status: {'🟢 Ready' if status.operational else '🔴 Not operational'}")
    print(f"Queue length: {status.pending_jobs}")
    print(f"Basis gates: {backend.configuration().basis_gates}")
    
    return service, backend

def get_ibm_primitives(session, **user_options):
    """
    Get IBM Quantum primitives (Sampler and Estimator) configured for the session.
    
    Args:
        session: Active IBM Quantum session
        **user_options: Additional options for the primitives
        
    Returns:
        Tuple of (Sampler, Estimator)
    """
    # Initialize primitives with the session
    sampler = Sampler(
        mode=session,
        options={
            "default_shots": user_options.get("shots", 1024),
            "dynamical_decoupling": {
                "enable": True,
                "sequence_type": "XpXm"
            }
        }
    )
    
    estimator = Estimator(
        mode=session,
        options={
            "default_shots": user_options.get("shots", 1024),
            "resilience_level": user_options.get("resilience_level", 1)
        }
    )
    
    print("\nConfigured for quantum hardware execution:")
    print(f"Default shots: {sampler.options.default_shots}")
    print(f"Dynamical decoupling: {sampler.options.dynamical_decoupling.sequence_type}")
    print(f"Resilience level: {estimator.options.resilience_level}")
    
    return sampler, estimator

def run_on_ibm(circuit, backend_name: str = "ibm_brisbane", **user_options):
    """
    Run a quantum circuit on an IBM Quantum backend.
    
    Args:
        circuit: The quantum circuit to run
        backend_name: Name of the IBM Quantum backend to use
        **user_options: Additional options for the execution
        
    Returns:
        The execution results
    """
    service, backend = get_ibm_backend(backend_name)
    
    # Create session and run within its context
    with Session(backend=backend) as session:
        # Initialize sampler with session
        sampler = Sampler(
            mode=session,
            options={
                "default_shots": user_options.get("shots", 1024),
                "dynamical_decoupling": {
                    "enable": True,
                    "sequence_type": "XpXm"
                }
            }
        )
        
        # Run the circuit
        print(f"\nSubmitting circuit to {backend.name}...")
        job = sampler.run([circuit])  # V2 format: list of circuits
        job_id = job.job_id()
        print(f"Job ID: {job_id}")
        print("Waiting for results...")
        result = job.result()
        
        # Get counts from the classical register "meas"
        counts = result[0].data.meas.get_counts()
        print(f"Counts: {counts}")
        
        return result 