"""
Quantum Database Schema Optimizer

This package provides tools for optimizing database schemas using quantum autoencoders.
It analyzes database structures, converts them to quantum states, and uses quantum
optimization to suggest improvements for query performance, storage efficiency,
and overall database design.
"""

from .schema.analyzer import SchemaAnalyzer
from .schema.graph import SchemaGraph
from .schema.metrics import SchemaMetrics
from .quantum.state import QuantumStateConverter
from .quantum.circuit import QuantumCircuitBuilder
from .quantum.optimizer import QuantumSchemaOptimizer

__version__ = '0.1.0'
__all__ = [
    'SchemaAnalyzer',
    'SchemaGraph',
    'SchemaMetrics',
    'QuantumStateConverter',
    'QuantumCircuitBuilder',
    'QuantumSchemaOptimizer'
] 