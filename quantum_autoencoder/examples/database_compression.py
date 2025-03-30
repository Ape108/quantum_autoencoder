"""
Database Compression using Quantum Autoencoder

This example demonstrates how to use the quantum autoencoder for compressing
database entries by converting them to quantum states and back.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import EfficientSU2
import pandas as pd
from pathlib import Path
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split, KFold
from scipy import stats

from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.core.training import train_autoencoder
from quantum_autoencoder.config.molecular_compression_config import MOLECULAR_COMPRESSION_CONFIG

def generate_molecular_features(n_samples: int = None) -> np.ndarray:
    """
    Generate molecular features using RDKit for testing.
    
    Args:
        n_samples: Number of molecular samples to generate (defaults to config value)
        
    Returns:
        Array of molecular features
    """
    config = MOLECULAR_COMPRESSION_CONFIG
    n_samples = n_samples or config["n_samples"]
    
    # List of molecular descriptors to calculate
    descriptors = [
        Descriptors.ExactMolWt,
        Descriptors.NumRotatableBonds,
        Descriptors.NumHAcceptors,
        Descriptors.NumHDonors,
        Descriptors.TPSA,
        Descriptors.MolLogP,
        Descriptors.MolMR,
        Descriptors.RingCount,
        Descriptors.NumAromaticRings,
        Descriptors.NumSaturatedRings,
        Descriptors.NumAliphaticRings,
        Descriptors.NumAromaticHeterocycles,
        Descriptors.NumSaturatedHeterocycles,
        Descriptors.NumAliphaticHeterocycles,
        Descriptors.NumAromaticCarbocycles
    ]
    
    # Generate features for each molecule
    features = []
    for _ in range(n_samples):
        # Randomly select a molecule from the list
        smiles = np.random.choice(config["smiles_list"])
        mol = Chem.MolFromSmiles(smiles)
        
        # Calculate descriptors
        mol_features = []
        for desc in descriptors:
            try:
                value = desc(mol)
                mol_features.append(value)
            except:
                mol_features.append(0.0)  # Fallback value if descriptor calculation fails
        
        features.append(mol_features)
    
    # Convert to numpy array and normalize
    features = np.array(features)
    
    # Handle zero standard deviation
    std = np.std(features, axis=0)
    std[std == 0] = 1.0  # Replace zero std with 1 to avoid division by zero
    
    # Normalize features
    features = (features - np.mean(features, axis=0)) / std
    
    return features

class DatabaseCompressor:
    """
    A class for compressing database entries using quantum autoencoder.
    """
    
    def __init__(
        self,
        n_features: int,
        n_latent: int,
        feature_encoding: str = "amplitude",
        reps: int = 3
    ):
        """
        Initialize the database compressor.
        
        Args:
            n_features: Number of features in each database entry
            n_latent: Number of qubits in latent space
            feature_encoding: Encoding method ('amplitude', 'basis', or 'angle')
            reps: Number of repetitions in the autoencoder circuit
        """
        self.n_features = n_features
        self.n_latent = n_latent
        self.feature_encoding = feature_encoding
        
        # Calculate required number of qubits
        if feature_encoding == "amplitude":
            self.n_qubits = max(int(np.ceil(np.log2(n_features))), n_latent + 1)
        else:
            self.n_qubits = max(n_features, n_latent + 1)
            
        # Initialize autoencoder
        self.autoencoder = QuantumAutoencoder(
            n_qubits=self.n_qubits,
            n_latent=n_latent,
            reps=reps
        )
        
        # Store optimal parameters after training
        self.optimal_params = None
        
    def encode_features(self, features: np.ndarray) -> QuantumCircuit:
        """
        Convert classical features to quantum state.
        
        Args:
            features: Array of classical features
            
        Returns:
            Quantum circuit representing the features
        """
        qc = QuantumCircuit(self.n_qubits)
        
        if self.feature_encoding == "amplitude":
            # Pad features with zeros if needed
            padded_features = np.zeros(2**self.n_qubits)
            padded_features[:len(features)] = features
            
            # Normalize features
            norm = np.sqrt(np.sum(np.abs(padded_features) ** 2))
            if norm > 1e-10:
                padded_features = padded_features / norm
                
            # Create state preparation circuit
            qc.initialize(padded_features, range(self.n_qubits))
            
        elif self.feature_encoding == "basis":
            # Convert features to binary and apply X gates
            for i, feature in enumerate(features):
                if i < self.n_qubits and feature > 0.5:  # Threshold for binary conversion
                    qc.x(i)
                    
        else:  # angle encoding
            # Use angle encoding for continuous features
            for i, feature in enumerate(features):
                if i < self.n_qubits:
                    qc.ry(feature * np.pi, i)
                
        return qc
    
    def decode_features(self, quantum_state: QuantumCircuit) -> np.ndarray:
        """
        Convert quantum state back to classical features.
        
        Args:
            quantum_state: Quantum circuit representing the state
            
        Returns:
            Array of classical features
        """
        # Get statevector
        sv = Statevector(quantum_state)
        
        if self.feature_encoding == "amplitude":
            # Return amplitudes directly
            return np.array(sv.data)
            
        elif self.feature_encoding == "basis":
            # Measure in computational basis
            counts = sv.probabilities_dict()
            features = np.zeros(self.n_features)
            for bitstring, prob in counts.items():
                idx = int(bitstring, 2)
                if idx < self.n_features:
                    features[idx] = prob
            return features
            
        else:  # angle encoding
            # Extract angles from statevector
            features = np.zeros(self.n_features)
            for i in range(self.n_features):
                # Estimate angle from statevector components
                features[i] = np.angle(sv.data[i]) / np.pi
            return features
    
    def compress_entry(self, entry: np.ndarray) -> Tuple[QuantumCircuit, np.ndarray]:
        """
        Compress a single database entry.
        
        Args:
            entry: Array of features for one database entry
            
        Returns:
            Tuple of (compressed quantum state, reconstruction error)
        """
        # Convert to quantum state
        qc = self.encode_features(entry)
        
        # Compress using autoencoder
        encoded = self.autoencoder.encode(qc, self.optimal_params)
        decoded = self.autoencoder.decode(encoded, self.optimal_params)
        
        # Calculate reconstruction error
        original_sv = Statevector(qc)
        reconstructed_sv = Statevector(decoded)
        error = 1 - np.abs(original_sv.inner(reconstructed_sv)) ** 2
        
        return encoded, error
    
    def train(self, training_data: List[np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Train the autoencoder on a dataset.
        
        Args:
            training_data: List of feature arrays
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training results
        """
        # Convert training data to quantum states
        training_states = [self.encode_features(entry) for entry in training_data]
        
        # Train on first state (can be extended to multiple states)
        optimal_params, final_cost = train_autoencoder(
            self.autoencoder,
            training_states[0],
            **kwargs
        )
        
        self.optimal_params = optimal_params
        
        # Calculate average reconstruction error
        errors = []
        for entry in training_data:
            _, error = self.compress_entry(entry)
            errors.append(error)
            
        return {
            "final_cost": final_cost,
            "fidelity": 1 - final_cost,
            "avg_reconstruction_error": np.mean(errors),
            "max_reconstruction_error": np.max(errors)
        }

def validate_reconstruction(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Validate reconstruction quality using multiple metrics.
    
    Args:
        original: Original feature array
        reconstructed: Reconstructed feature array (may be complex)
        
    Returns:
        Dictionary of validation metrics
    """
    # Ensure arrays are 2D and convert complex to real by taking magnitude
    original = np.asarray(original).reshape(-1, original.shape[-1])
    reconstructed = np.abs(np.asarray(reconstructed)).reshape(-1, reconstructed.shape[-1])
    
    # Calculate basic statistics
    metrics = {
        'mse': np.mean((original - reconstructed) ** 2),
        'mae': np.mean(np.abs(original - reconstructed)),
        'max_error': np.max(np.abs(original - reconstructed)),
        'mean_std_ratio': np.std(reconstructed) / np.std(original),
    }
    
    # Calculate correlation metrics only if we have multiple samples
    if original.shape[0] > 1:
        metrics['r2'] = stats.pearsonr(original.flatten(), reconstructed.flatten())[0]**2
        metrics['feature_correlations'] = np.corrcoef(original.flatten(), reconstructed.flatten())[0,1]
    else:
        metrics['r2'] = np.nan
        metrics['feature_correlations'] = np.nan
    
    # Calculate per-feature errors
    feature_errors = np.abs(original - reconstructed)
    metrics['feature_errors'] = {
        'mean': np.mean(feature_errors, axis=0),
        'std': np.std(feature_errors, axis=0),
        'max': np.max(feature_errors, axis=0)
    }
    
    return metrics

def example_usage():
    """Example usage of the DatabaseCompressor with molecular features."""
    config = MOLECULAR_COMPRESSION_CONFIG
    
    # Generate molecular features using RDKit
    features = generate_molecular_features()
    
    print(f"Generated {len(features)} molecular samples with {features.shape[1]} features")
    print("\nFeature descriptions:")
    for i, desc in enumerate(config["feature_descriptions"], 1):
        print(f"{i}. {desc}")
    
    # Split into training and test sets
    train_features, test_features = train_test_split(
        features, 
        test_size=config["test_size"], 
        random_state=config["random_state"]
    )
    
    print(f"\nTesting compression with {config['n_latent']} latent qubits...")
    
    # Initialize compressor
    compressor = DatabaseCompressor(
        n_features=features.shape[1],
        n_latent=config["n_latent"],
        feature_encoding=config["feature_encoding"],
        reps=config["reps"]
    )
    
    # Perform k-fold cross-validation
    kf = KFold(n_splits=config["n_folds"], shuffle=True, random_state=config["random_state"])
    
    cv_errors = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
        print(f"\nFold {fold + 1}/{config['n_folds']}")
        fold_train = train_features[train_idx]
        fold_val = train_features[val_idx]
        
        # Train on fold
        results = compressor.train(
            fold_train,
            maxiter=config["maxiter"],
            n_trials=config["n_trials"],
            optimizer=config["optimizer"],
            options=config["options"]
        )
        
        # Early stopping if we achieve target fidelity
        if results['fidelity'] > config["fidelity_threshold"]:
            print(f"Achieved {results['fidelity']:.4f} fidelity, stopping early")
            break
        
        # Validate on fold
        val_errors = []
        for entry in fold_val:
            _, error = compressor.compress_entry(entry)
            val_errors.append(error)
        
        cv_errors.extend(val_errors)
        print(f"Fold {fold + 1} validation error: {np.mean(val_errors):.4f}")
    
    mean_cv_error = np.mean(cv_errors)
    print(f"\nCross-validation results:")
    print(f"Mean CV error: {mean_cv_error:.4f}")
    print(f"Std CV error: {np.std(cv_errors):.4f}")
    
    print("\nConfiguration details:")
    print(f"Number of latent qubits: {config['n_latent']}")
    print(f"Compression ratio: {(features.shape[1] - config['n_latent'])/features.shape[1]*100:.1f}%")
    print(f"Final fidelity: {results['fidelity']:.4f}")
    print(f"Average reconstruction error: {results['avg_reconstruction_error']:.4f}")
    
    # Test on holdout set
    print("\nTesting on holdout set...")
    holdout_errors = []
    holdout_reconstructions = []
    
    for i, molecule in enumerate(test_features):
        compressed_state, error = compressor.compress_entry(molecule)
        reconstructed = compressor.decode_features(compressed_state)
        holdout_errors.append(error)
        holdout_reconstructions.append(reconstructed[:len(molecule)])  # Trim to actual features
    
    holdout_errors = np.array(holdout_errors)
    holdout_reconstructions = np.array(holdout_reconstructions)
    
    print(f"Holdout set results:")
    print(f"Mean reconstruction error: {np.mean(holdout_errors):.4f}")
    print(f"Max reconstruction error: {np.max(holdout_errors):.4f}")
    print(f"Min reconstruction error: {np.min(holdout_errors):.4f}")
    print(f"Std reconstruction error: {np.std(holdout_errors):.4f}")
    
    # Validate reconstruction quality
    validation_metrics = validate_reconstruction(test_features, holdout_reconstructions)
    print("\nReconstruction validation metrics:")
    print(f"MSE: {validation_metrics['mse']:.6f}")
    print(f"MAE: {validation_metrics['mae']:.6f}")
    if not np.isnan(validation_metrics['r2']):
        print(f"R²: {validation_metrics['r2']:.6f}")
    print(f"Max error: {validation_metrics['max_error']:.6f}")
    print(f"Mean/Std ratio: {validation_metrics['mean_std_ratio']:.6f}")
    if not np.isnan(validation_metrics['feature_correlations']):
        print(f"Feature correlations: {validation_metrics['feature_correlations']:.6f}")
    
    # Show detailed reconstruction for best case
    best_idx = np.argmin(holdout_errors)
    test_molecule = test_features[best_idx]
    reconstructed = holdout_reconstructions[best_idx]
    
    print(f"\nBest Case Reconstruction (Molecule {best_idx}):")
    print(f"Reconstruction error: {holdout_errors[best_idx]:.4f}")
    print("\nOriginal vs Reconstructed Properties (showing magnitude of complex numbers):")
    for i, (orig, rec) in enumerate(zip(test_molecule, reconstructed)):
        magnitude = np.abs(rec)
        phase = np.angle(rec) if np.abs(rec) > 1e-10 else 0
        print(f"Property {i+1}: {orig:.4f} -> |{magnitude:.4f}|∠{phase:.4f}")
        
    # Show detailed reconstruction for worst case
    worst_idx = np.argmax(holdout_errors)
    test_molecule = test_features[worst_idx]
    reconstructed = holdout_reconstructions[worst_idx]
    
    print(f"\nWorst Case Reconstruction (Molecule {worst_idx}):")
    print(f"Reconstruction error: {holdout_errors[worst_idx]:.4f}")
    print("\nOriginal vs Reconstructed Properties (showing magnitude of complex numbers):")
    for i, (orig, rec) in enumerate(zip(test_molecule, reconstructed)):
        magnitude = np.abs(rec)
        phase = np.angle(rec) if np.abs(rec) > 1e-10 else 0
        print(f"Property {i+1}: {orig:.4f} -> |{magnitude:.4f}|∠{phase:.4f}")

def proof_of_concept():
    """Quick proof of concept test with minimal dataset and iterations."""
    print("Running proof of concept test...")
    
    config = MOLECULAR_COMPRESSION_CONFIG
    
    # Generate a small dataset
    features = generate_molecular_features(n_samples=5)
    print(f"\nGenerated {len(features)} molecular samples with {features.shape[1]} features")
    
    # Initialize compressor with minimal configuration
    compressor = DatabaseCompressor(
        n_features=features.shape[1],
        n_latent=4,  # Use fewer qubits for quick test
        feature_encoding=config["feature_encoding"],
        reps=2  # Use fewer repetitions for quick test
    )
    
    # Train with minimal iterations
    print("\nTraining with minimal iterations...")
    results = compressor.train(
        features,
        maxiter=50,  # Reduced from config value
        n_trials=2,  # Reduced from config value
        optimizer=config["optimizer"],
        options={"shots": 256}  # Reduced from config value
    )
    
    print("\nTraining results:")
    print(f"Final cost: {results['final_cost']:.4f}")
    print(f"Fidelity: {results['fidelity']:.4f}")
    print(f"Average reconstruction error: {results['avg_reconstruction_error']:.4f}")
    
    # Test reconstruction on all molecules
    print("\nTesting reconstruction on all molecules...")
    all_errors = []
    all_reconstructions = []
    
    for i, molecule in enumerate(features):
        compressed_state, error = compressor.compress_entry(molecule)
        reconstructed = compressor.decode_features(compressed_state)
        all_errors.append(error)
        all_reconstructions.append(reconstructed[:len(molecule)])
        
        print(f"\nMolecule {i+1}:")
        print(f"Reconstruction error: {error:.4f}")
        print("Original vs Reconstructed Properties (showing magnitude of complex numbers):")
        for j, (orig, rec) in enumerate(zip(molecule, reconstructed[:len(molecule)])):
            magnitude = np.abs(rec)
            phase = np.angle(rec) if np.abs(rec) > 1e-10 else 0
            print(f"Property {j+1}: {orig:.4f} -> |{magnitude:.4f}|∠{phase:.4f}")
    
    all_errors = np.array(all_errors)
    all_reconstructions = np.array(all_reconstructions)
    
    print("\nOverall results:")
    print(f"Mean reconstruction error: {np.mean(all_errors):.4f}")
    print(f"Max reconstruction error: {np.max(all_errors):.4f}")
    print(f"Min reconstruction error: {np.min(all_errors):.4f}")
    print(f"Std reconstruction error: {np.std(all_errors):.4f}")
    
    # Validate reconstruction quality
    validation_metrics = validate_reconstruction(features, all_reconstructions)
    
    print("\nReconstruction validation metrics:")
    print(f"MSE: {validation_metrics['mse']:.6f}")
    print(f"MAE: {validation_metrics['mae']:.6f}")
    if not np.isnan(validation_metrics['r2']):
        print(f"R²: {validation_metrics['r2']:.6f}")
    print(f"Max error: {validation_metrics['max_error']:.6f}")
    print(f"Mean/Std ratio: {validation_metrics['mean_std_ratio']:.6f}")
    if not np.isnan(validation_metrics['feature_correlations']):
        print(f"Feature correlations: {validation_metrics['feature_correlations']:.6f}")
    
    print("\nPer-feature errors:")
    for i, (mean, std, max_err) in enumerate(zip(
        validation_metrics['feature_errors']['mean'],
        validation_metrics['feature_errors']['std'],
        validation_metrics['feature_errors']['max']
    )):
        print(f"Feature {i+1}: mean={mean:.4f}, std={std:.4f}, max={max_err:.4f}")
    
    # Ask if user wants to proceed with full test
    print("\nProof of concept completed. The results show:")
    print(f"1. Training fidelity: {results['fidelity']:.4f}")
    print(f"2. Average reconstruction error: {results['avg_reconstruction_error']:.4f}")
    print(f"3. Best case error: {np.min(all_errors):.4f}")
    print(f"4. Worst case error: {np.max(all_errors):.4f}")
    print(f"5. MSE: {validation_metrics['mse']:.4f}")
    print(f"6. MAE: {validation_metrics['mae']:.4f}")
    
    print("\nConfiguration for full test:")
    print(f"1. Number of latent qubits: {config['n_latent']}")
    print(f"2. Circuit repetitions: {config['reps']}")
    print(f"3. Training iterations: {config['maxiter']}")
    print(f"4. Number of trials: {config['n_trials']}")
    print(f"5. Target fidelity: {config['fidelity_threshold']}")

if __name__ == "__main__":
    # Run proof of concept first
    proof_of_concept()
    
    # Ask user if they want to run the full test
    print("\nBased on these results, would you like to:")
    print("1. Run the full test with current parameters")
    print("2. Adjust parameters and try again")
    print("3. Exit")
    response = input("Enter your choice (1/2/3): ")
    
    if response == "1":
        example_usage()
    elif response == "2":
        print("\nSuggested parameter adjustments:")
        print("1. Increase number of latent qubits")
        print("2. Increase number of circuit repetitions")
        print("3. Try different encoding method")
        print("4. Increase training iterations")
        print("Please modify the code with your desired changes and run again.") 