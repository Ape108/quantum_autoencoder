# Quantum Database Optimizer

A quantum computing-based tool for optimizing database structure and performance using quantum autoencoders.

## Overview

This project uses quantum autoencoders to analyze and optimize database structures. By encoding database features into quantum states and finding efficient latent representations, it can identify patterns and suggest optimizations for improved database performance.

## Features

- Quantum feature extraction from database schemas and queries
- Quantum autoencoder-based pattern detection
- Database structure optimization
- SQL export and analysis tools
- Comprehensive results formatting and visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python -m quantum_autoencoder.database_optimization.quantum.main <database_path> --output-dir results --n-qubits 4 --n-latent 2
```

This will:
1. Analyze the input database
2. Train a quantum autoencoder
3. Generate optimized database
4. Create detailed reports and visualizations

## Results

The optimization process generates:
- Optimized SQLite database
- SQL exports of original and optimized databases
- Compression metrics and analysis
- Human-readable summary report
- Visualization of database structure and optimizations

Results are organized in the following structure:
```
results/
├── SUMMARY.md           # Human-readable summary
├── metrics/            
│   └── compression_metrics.json  # Detailed metrics
├── sql/
│   ├── original_schema.sql      # Original DB schema
│   ├── original_data.sql        # Original DB data
│   ├── optimized_schema.sql     # Optimized DB schema
│   └── optimized_data.sql       # Optimized DB data
└── optimized.db        # SQLite database file
```

## Example

The `tests/database_optimization/ExampleDB` directory contains a sample Northwind database that can be used to test the optimization process:

```bash
python -m quantum_autoencoder.database_optimization.quantum.main tests/database_optimization/ExampleDB/northwind.db
```

## License

MIT License - see LICENSE file for details 