from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
with open('README.md') as f:
    long_description = f.read()

setup(
    name="quantum_autoencoder",
    version="0.1.0",
    description="Quantum autoencoder implementation using Qiskit V2 primitives",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "qiskit>=1.0.0",
        "qiskit-machine-learning>=0.8.2",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Quantum Computing",
        "Framework :: Qiskit",
    ],
    keywords="quantum computing, machine learning, autoencoder, qiskit, v2 primitives, error mitigation",
    project_urls={
        "Documentation": "https://docs.quantum.ibm.com/api/migration-guides/v2-primitives",
        "Source": "https://github.com/Ape108/quantum_autoencoder",
        "Bug Reports": "https://github.com/Ape108/quantum_autoencoder/issues",
    }
) 