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
    description="A quantum autoencoder implementation using Qiskit V2 primitives",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Cameron Akhtar",
    author_email="your.email@example.com",  # Replace with your email
    url="https://github.com/Ape108/quantum_autoencoder",
    packages=find_packages(),
    install_requires=requirements,
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
    python_requires=">=3.9",  # Based on latest Qiskit requirements
    keywords="quantum computing, machine learning, autoencoder, qiskit, v2 primitives, error mitigation",
    project_urls={
        "Documentation": "https://docs.quantum.ibm.com/api/migration-guides/v2-primitives",
        "Source": "https://github.com/Ape108/quantum_autoencoder",
        "Bug Reports": "https://github.com/Ape108/quantum_autoencoder/issues",
    }
) 