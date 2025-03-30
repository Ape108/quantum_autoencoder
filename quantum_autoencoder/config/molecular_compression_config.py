"""
Configuration for molecular feature compression using quantum autoencoder.
"""

MOLECULAR_COMPRESSION_CONFIG = {
    # Dataset configuration
    "n_samples": 50,
    "test_size": 0.2,
    "random_state": 42,
    
    # Feature generation
    "smiles_list": [
        'CC(=O)O',  # Acetic acid
        'CCO',      # Ethanol
        'CCCC',     # Butane
        'c1ccccc1', # Benzene
        'CC(=O)N',  # Acetamide
        'CCN',      # Ethylamine
        'CCOC',     # Dimethyl ether
        'CC=O',     # Acetaldehyde
        'CC#N',     # Acetonitrile
        'CS',       # Methyl mercaptan
    ],
    
    # Quantum autoencoder configuration
    "n_latent": 8,          # Number of latent qubits
    "reps": 5,              # Number of circuit repetitions
    "feature_encoding": "amplitude",
    
    # Training configuration
    "maxiter": 1000,
    "n_trials": 5,
    "optimizer": "COBYLA",
    "options": {
        "shots": 1024
    },
    
    # Cross-validation
    "n_folds": 3,
    
    # Early stopping
    "fidelity_threshold": 0.99,
    
    # Feature descriptions
    "feature_descriptions": [
        "Exact Molecular Weight",
        "Number of Rotatable Bonds",
        "Number of H Acceptors",
        "Number of H Donors",
        "Topological Polar Surface Area (TPSA)",
        "LogP",
        "Molar Refractivity",
        "Ring Count",
        "Number of Aromatic Rings",
        "Number of Saturated Rings",
        "Number of Aliphatic Rings",
        "Number of Aromatic Heterocycles",
        "Number of Saturated Heterocycles",
        "Number of Aliphatic Heterocycles",
        "Number of Aromatic Carbocycles"
    ]
} 