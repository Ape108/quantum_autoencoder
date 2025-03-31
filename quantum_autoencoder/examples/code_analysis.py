"""
Analysis and visualization tools for code compression results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CompressionMetrics:
    """Container for compression metrics."""
    file_name: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    reconstruction_error: float
    language: str
    timestamp: str

class CodeCompressionAnalyzer:
    def __init__(self, results_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Directory containing compression results
        """
        self.results_dir = Path(results_dir)
        self.metrics: List[CompressionMetrics] = []
        
    def add_metrics(self, metrics: CompressionMetrics):
        """Add compression metrics to the analysis."""
        self.metrics.append(metrics)
        
    def _get_language_from_extension(self, file_name: str) -> str:
        """Determine programming language from file extension."""
        ext = os.path.splitext(file_name)[1].lower()
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C++',
            '.hpp': 'C++',
            '.cs': 'C#',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.go': 'Go',
            '.rs': 'Rust',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala'
        }
        return language_map.get(ext, 'Unknown')
    
    def plot_compression_ratios(self, save_path: str = None):
        """Plot compression ratios by language."""
        plt.figure(figsize=(12, 6))
        
        # Group metrics by language
        language_metrics = {}
        for metric in self.metrics:
            if metric.language not in language_metrics:
                language_metrics[metric.language] = []
            language_metrics[metric.language].append(metric.compression_ratio)
        
        # Create box plot
        data = [ratios for ratios in language_metrics.values()]
        labels = list(language_metrics.keys())
        
        plt.boxplot(data, labels=labels)
        plt.title('Code Compression Ratios by Language')
        plt.ylabel('Compression Ratio (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_error_distribution(self, save_path: str = None):
        """Plot reconstruction error distribution."""
        plt.figure(figsize=(12, 6))
        
        # Group errors by language
        language_errors = {}
        for metric in self.metrics:
            if metric.language not in language_errors:
                language_errors[metric.language] = []
            language_errors[metric.language].append(metric.reconstruction_error)
        
        # Create violin plot
        data = [errors for errors in language_errors.values()]
        labels = list(language_errors.keys())
        
        plt.violinplot(data, labels=labels)
        plt.title('Reconstruction Error Distribution by Language')
        plt.ylabel('Reconstruction Error')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_size_comparison(self, save_path: str = None):
        """Plot original vs compressed sizes."""
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        original_sizes = [metric.original_size for metric in self.metrics]
        compressed_sizes = [metric.compressed_size for metric in self.metrics]
        languages = [metric.language for metric in self.metrics]
        
        # Create scatter plot
        plt.scatter(original_sizes, compressed_sizes, c=range(len(languages)), cmap='tab10')
        
        # Add labels for each point
        for i, (orig, comp, lang) in enumerate(zip(original_sizes, compressed_sizes, languages)):
            plt.annotate(lang, (orig, comp), xytext=(5, 5), textcoords='offset points')
        
        plt.title('Original vs Compressed File Sizes')
        plt.xlabel('Original Size (bytes)')
        plt.ylabel('Compressed Size (bytes)')
        plt.grid(True, alpha=0.3)
        
        # Add diagonal line
        max_size = max(max(original_sizes), max(compressed_sizes))
        plt.plot([0, max_size], [0, max_size], 'k--', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def generate_report(self, output_dir: str = None):
        """
        Generate a comprehensive analysis report.
        
        Args:
            output_dir: Directory to save the report and plots
        """
        if output_dir is None:
            output_dir = self.results_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.plot_compression_ratios(str(output_dir / f'compression_ratios_{timestamp}.png'))
        self.plot_error_distribution(str(output_dir / f'error_distribution_{timestamp}.png'))
        self.plot_size_comparison(str(output_dir / f'size_comparison_{timestamp}.png'))
        
        # Generate summary statistics
        summary = {
            'total_files': len(self.metrics),
            'languages': set(metric.language for metric in self.metrics),
            'avg_compression_ratio': np.mean([metric.compression_ratio for metric in self.metrics]),
            'avg_reconstruction_error': np.mean([metric.reconstruction_error for metric in self.metrics]),
            'best_compression': max(self.metrics, key=lambda x: x.compression_ratio),
            'worst_compression': min(self.metrics, key=lambda x: x.compression_ratio),
            'best_error': min(self.metrics, key=lambda x: x.reconstruction_error),
            'worst_error': max(self.metrics, key=lambda x: x.reconstruction_error)
        }
        
        # Save summary to file
        with open(output_dir / f'summary_{timestamp}.txt', 'w') as f:
            f.write("Code Compression Analysis Summary\n")
            f.write("================================\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Files Analyzed: {summary['total_files']}\n")
            f.write(f"Languages: {', '.join(summary['languages'])}\n\n")
            
            f.write("Overall Statistics:\n")
            f.write(f"Average Compression Ratio: {summary['avg_compression_ratio']:.2f}%\n")
            f.write(f"Average Reconstruction Error: {summary['avg_reconstruction_error']:.4f}\n\n")
            
            f.write("Best Compression:\n")
            f.write(f"File: {summary['best_compression'].file_name}\n")
            f.write(f"Language: {summary['best_compression'].language}\n")
            f.write(f"Compression Ratio: {summary['best_compression'].compression_ratio:.2f}%\n")
            f.write(f"Reconstruction Error: {summary['best_compression'].reconstruction_error:.4f}\n\n")
            
            f.write("Worst Compression:\n")
            f.write(f"File: {summary['worst_compression'].file_name}\n")
            f.write(f"Language: {summary['worst_compression'].language}\n")
            f.write(f"Compression Ratio: {summary['worst_compression'].compression_ratio:.2f}%\n")
            f.write(f"Reconstruction Error: {summary['worst_compression'].reconstruction_error:.4f}\n\n")
            
            f.write("Best Reconstruction:\n")
            f.write(f"File: {summary['best_error'].file_name}\n")
            f.write(f"Language: {summary['best_error'].language}\n")
            f.write(f"Reconstruction Error: {summary['best_error'].reconstruction_error:.4f}\n\n")
            
            f.write("Worst Reconstruction:\n")
            f.write(f"File: {summary['worst_error'].file_name}\n")
            f.write(f"Language: {summary['worst_error'].language}\n")
            f.write(f"Reconstruction Error: {summary['worst_error'].reconstruction_error:.4f}\n\n")
            
            f.write("Detailed Results:\n")
            f.write("----------------\n")
            for metric in sorted(self.metrics, key=lambda x: x.compression_ratio, reverse=True):
                f.write(f"\n{metric.file_name} ({metric.language}):\n")
                f.write(f"Original Size: {metric.original_size} bytes\n")
                f.write(f"Compressed Size: {metric.compressed_size} bytes\n")
                f.write(f"Compression Ratio: {metric.compression_ratio:.2f}%\n")
                f.write(f"Reconstruction Error: {metric.reconstruction_error:.4f}\n") 