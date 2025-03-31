"""
Schema analysis and representation module.

This module provides tools for analyzing and representing database schemas
as graph structures, including metrics calculation and feature extraction.
"""

from .analyzer import SchemaAnalyzer
from .graph import SchemaGraph
from .metrics import SchemaMetrics

__all__ = ['SchemaAnalyzer', 'SchemaGraph', 'SchemaMetrics'] 