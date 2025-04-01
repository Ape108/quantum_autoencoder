from setuptools import setup, find_packages

setup(
    name="quantum-db-optimizer",
    version="0.1.0",
    description="Quantum computing-based database optimization tool",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "qiskit>=1.0.0",
        "matplotlib>=3.4.0",
        "tabulate>=0.8.0",
        "networkx>=2.6.0"
    ],
    entry_points={
        "console_scripts": [
            "quantum-db-optimizer=quantum_db_optimizer.__main__:main"
        ]
    },
    python_requires=">=3.8"
) 