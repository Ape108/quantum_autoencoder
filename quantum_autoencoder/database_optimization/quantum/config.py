class QuantumConfig:
    n_qubits = 2  # Absolute minimum
    n_latent = 1  # Minimum compression
    n_epochs = 5  # Very few epochs
    learning_rate = 0.5  # Aggressive learning
    batch_size = 8  # Larger batches
    optimizer = "adam"
    early_stopping_patience = 2
    early_stopping_threshold = 0.01  # Less strict threshold
    # ... existing code ... 