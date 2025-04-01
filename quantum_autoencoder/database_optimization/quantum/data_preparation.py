import numpy as np
from typing import List, Tuple

def prepare_quantum_data(features: List[float], n_qubits: int) -> np.ndarray:
    """Prepare classical data for quantum processing."""
    # Normalize features to [0, 1]
    features = np.array(features)
    if len(features) > 0:
        features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
    
    # Pad or truncate to match n_qubits
    if len(features) < n_qubits:
        features = np.pad(features, (0, n_qubits - len(features)))
    else:
        features = features[:n_qubits]
    
    # Convert to quantum amplitudes
    amplitudes = np.sqrt(features)
    # Normalize
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        amplitudes = amplitudes / norm
    
    return amplitudes

def encode_database_features(database_features: List[List[float]], n_qubits: int) -> List[np.ndarray]:
    """Encode database features into quantum states."""
    quantum_states = []
    for features in database_features:
        quantum_state = prepare_quantum_data(features, n_qubits)
        quantum_states.append(quantum_state)
    return quantum_states 