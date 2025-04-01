"""Generate quantum autoencoder architecture diagram."""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt

def create_architecture_diagram():
    # Create registers
    qr_input = QuantumRegister(2, 'in')
    qr_latent = QuantumRegister(1, 'lat')
    qr_trash = QuantumRegister(1, 'tr')
    cr = ClassicalRegister(1, 'c')
    
    # Create circuit
    qc = QuantumCircuit(qr_input, qr_latent, qr_trash, cr)
    
    # Add encoder U
    qc.barrier()
    qc.ry(np.pi/4, qr_input[0])
    qc.ry(np.pi/4, qr_input[1])
    qc.cx(qr_input[0], qr_input[1])
    qc.barrier()
    
    # Add encoder V
    qc.ry(np.pi/4, qr_latent)
    qc.ry(np.pi/4, qr_trash)
    qc.cx(qr_latent, qr_trash)
    qc.barrier()
    
    # Add measurement
    qc.measure(qr_trash, cr)
    
    # Draw circuit
    fig = qc.draw('mpl', 
                  style={'backgroundcolor': '#FFFFFF'},
                  fold=20,  # Prevent long circuit
                  plot_barriers=True,
                  initial_state=True)
    plt.title('Quantum Autoencoder Architecture\n(Simplified Example)')
    plt.tight_layout()
    plt.savefig('docs/images/architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    create_architecture_diagram() 