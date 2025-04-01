"""
Generate visualizations for database compression results.
"""

import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

def create_schema_graph(db_path: str, output_path: str):
    """Create a visualization of the database schema and relationships."""
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get tables
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """)
    tables = [table[0] for table in cursor.fetchall()]
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes for tables
    for table in tables:
        # Get column info
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        # Create label with columns
        label = f"{table}\n"
        label += "\n".join([f"- {col[1]}" for col in columns])
        
        G.add_node(table, label=label)
        
        # Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list({table})")
        foreign_keys = cursor.fetchall()
        
        # Add edges for relationships
        for fk in foreign_keys:
            G.add_edge(table, fk[2], label=f"{fk[3]} → {fk[4]}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.7)
    nx.draw_networkx_labels(G, pos, 
                          labels=nx.get_node_attributes(G, 'label'),
                          font_size=8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    plt.title("Database Schema Relationships")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    conn.close()

def plot_compression_heatmap(metrics_path: str, output_path: str):
    """Create a heatmap showing the compression of different database aspects."""
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Extract schema info
    schema = metrics['database_info']['schema']
    
    # Create data matrix
    tables = list(schema.keys())
    aspects = ['Columns', 'Rows', 'Indexes', 'Foreign Keys']
    data = np.zeros((len(tables), len(aspects)))
    
    for i, table in enumerate(tables):
        data[i, 0] = len(schema[table]['columns'])
        data[i, 1] = schema[table]['row_count']
        data[i, 2] = len(schema[table]['indexes'])
        data[i, 3] = len(schema[table]['foreign_keys'])
    
    # Normalize data
    data = data / data.max(axis=0)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(data, aspect='auto', cmap='YlOrRd')
    
    # Add labels
    plt.xticks(range(len(aspects)), aspects, rotation=45)
    plt.yticks(range(len(tables)), tables)
    
    # Add colorbar
    plt.colorbar(label='Normalized Value')
    
    plt.title('Database Structure Heatmap')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_query_comparison(metrics_path: str, output_path: str):
    """Create a detailed visualization of query performance improvements."""
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    query_metrics = metrics['query_performance']
    queries = list(query_metrics.keys())
    
    # Extract metrics
    orig_times = [m['original_time'] for m in query_metrics.values()]
    opt_times = [m['optimized_time'] for m in query_metrics.values()]
    speedups = [m['speedup'] for m in query_metrics.values()]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot execution times
    x = np.arange(len(queries))
    width = 0.35
    
    ax1.bar(x - width/2, orig_times, width, label='Original', color='lightcoral')
    ax1.bar(x + width/2, opt_times, width, label='Optimized', color='lightgreen')
    
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Query Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(queries)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(orig_times):
        ax1.text(i - width/2, v, f'{v:.2e}s', ha='center', va='bottom')
    for i, v in enumerate(opt_times):
        ax1.text(i + width/2, v, f'{v:.2e}s', ha='center', va='bottom')
    
    # Plot speedup factors
    ax2.bar(x, speedups, color='lightblue')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Query Performance Improvement')
    ax2.set_xticks(x)
    ax2.set_xticklabels(queries)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(speedups):
        ax2.text(i, v, f'{v:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    """Generate all visualizations."""
    # Setup paths
    base_dir = Path("tests/database_optimization/ExampleDB")
    output_dir = Path("test_results/compression_test_20240331")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate schema visualization
    create_schema_graph(
        str(base_dir / "northwind.db"),
        str(output_dir / "schema_relationships.png")
    )
    
    # Generate compression heatmap
    plot_compression_heatmap(
        str(base_dir / "compression_metrics.json"),
        str(output_dir / "compression_heatmap.png")
    )
    
    # Generate detailed query comparison
    plot_query_comparison(
        str(base_dir / "compression_metrics.json"),
        str(output_dir / "query_analysis.png")
    )
    
    print("Visualizations generated successfully!")

if __name__ == "__main__":
    main() 