"""
Complete Database Compression Workflow

This script:
1. Loads an SQL database
2. Converts it to quantum states
3. Compresses using quantum autoencoder
4. Reconstructs and saves the optimized database
5. Provides comparison metrics and analysis tools
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from qiskit.quantum_info import state_fidelity
from qiskit import QuantumCircuit
from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.core.training import train_autoencoder
from quantum_autoencoder.database_optimization.schema.graph import SchemaGraph

class DatabaseCompressionWorkflow:
    def __init__(self, 
                 original_db_path: str,
                 compressed_db_path: str,
                 reconstructed_db_path: str,
                 metrics_path: str):
        """Initialize the workflow with file paths."""
        self.original_db_path = original_db_path
        self.compressed_db_path = compressed_db_path
        self.reconstructed_db_path = reconstructed_db_path
        self.metrics_path = metrics_path
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "database_info": {},
            "compression_metrics": {},
            "query_performance": {},
            "storage_metrics": {}
        }
    
    def extract_schema_info(self) -> Tuple[List[str], Dict]:
        """Extract schema information from original database."""
        conn = sqlite3.connect(self.original_db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = [table[0] for table in cursor.fetchall()]
        
        schema_info = {}
        for table in tables:
            # Get column info
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            # Get index info
            cursor.execute(f"PRAGMA index_list({table})")
            indexes = cursor.fetchall()
            
            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            foreign_keys = cursor.fetchall()
            
            schema_info[table] = {
                "columns": [col[1] for col in columns],
                "row_count": row_count,
                "indexes": [idx[1] for idx in indexes],
                "foreign_keys": [(fk[3], fk[4]) for fk in foreign_keys]  # (table, column)
            }
        
        conn.close()
        return tables, schema_info
    
    def prepare_quantum_features(self, schema_info: Dict) -> np.ndarray:
        """Convert schema information to quantum features."""
        features = []
        
        # Calculate normalization factors
        max_rows = max((table["row_count"] for table in schema_info.values()), default=1)
        max_cols = max((len(table["columns"]) for table in schema_info.values()), default=1)
        max_indexes = max((len(table["indexes"]) for table in schema_info.values()), default=1)
        max_fks = max((len(table["foreign_keys"]) for table in schema_info.values()), default=1)
        
        # Extract normalized features for each table
        for table_info in schema_info.values():
            features.extend([
                table_info["row_count"] / max_rows if max_rows > 0 else 0,
                len(table_info["columns"]) / max_cols if max_cols > 0 else 0,
                len(table_info["indexes"]) / max_indexes if max_indexes > 0 else 0,
                len(table_info["foreign_keys"]) / max_fks if max_fks > 0 else 0
            ])
        
        return np.array(features)
    
    def compress_database(self, features: np.ndarray, n_latent: int) -> Dict:
        """Compress database features using quantum autoencoder."""
        n_features = len(features)
        n_qubits = int(np.ceil(np.log2(n_features)))
        
        print(f"\nCompression Configuration:")
        print(f"- Features: {n_features}")
        print(f"- Input qubits: {n_qubits}")
        print(f"- Latent qubits: {n_latent}")
        print(f"- Compression ratio: {2**n_qubits}/{2**n_latent} = {2**(n_qubits-n_latent)}:1")
        
        # Initialize quantum autoencoder
        autoencoder = QuantumAutoencoder(
            n_qubits=n_qubits,
            n_latent=n_latent,
            reps=3,
            options={
                "optimization_level": 3,
                "resilience_level": 1,
                "shots": 1024,
                "dynamical_decoupling": {"enable": True}
            }
        )
        
        # Create input circuit
        norm_features = features / np.linalg.norm(features)
        input_circuit = QuantumCircuit(n_qubits)
        for i, amplitude in enumerate(norm_features):
            if i < n_qubits:
                input_circuit.ry(2 * np.arccos(amplitude), i)
        
        # Train autoencoder
        best_params, final_cost = train_autoencoder(
            autoencoder,
            input_circuit,
            maxiter=200,
            n_trials=5,
            plot_progress=True,
            save_dir="visualizations"
        )
        
        # Get final compression results
        encoded_state = autoencoder.encode(input_circuit, best_params)
        decoded_state = autoencoder.decode(encoded_state, best_params)
        final_fidelity = autoencoder.get_fidelity(input_circuit, decoded_state)
        
        return {
            "compressed_state": encoded_state,
            "reconstructed_state": decoded_state,
            "fidelity": final_fidelity,
            "parameters": best_params
        }
    
    def reconstruct_database(self, 
                           tables: List[str], 
                           schema_info: Dict, 
                           compression_results: Dict) -> None:
        """Reconstruct and optimize database from compressed state."""
        # Create new database
        conn = sqlite3.connect(self.reconstructed_db_path)
        cursor = conn.cursor()
        
        # Get original database connection for schema info
        orig_conn = sqlite3.connect(self.original_db_path)
        orig_cursor = orig_conn.cursor()
        
        # Drop existing tables if they exist
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        
        # Recreate schema with optimizations
        for table in tables:
            columns = schema_info[table]["columns"]
            col_defs = []
            
            # Get column types from original database
            orig_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
            create_stmt = orig_cursor.fetchone()[0]
            
            # Extract column definitions from CREATE statement
            col_start = create_stmt.find('(') + 1
            col_end = create_stmt.rfind(')')
            col_defs_str = create_stmt[col_start:col_end].strip()
            col_parts = [part.strip() for part in col_defs_str.split(',')]
            
            # Create mapping of column names to their full definitions
            col_map = {}
            for part in col_parts:
                if part:
                    words = part.split()
                    if words:
                        col_name = words[0].strip('"\'')
                        col_map[col_name] = part
            
            # Use the full column definitions from the original table
            col_defs = []
            for col in columns:
                if col in col_map:
                    col_defs.append(col_map[col])
                else:
                    # Fallback to TEXT type if we can't find the original definition
                    col_defs.append(f"{col} TEXT")
            
            # Create table
            cursor.execute(f"""
                CREATE TABLE {table} (
                    {', '.join(col_defs)}
                )
            """)
            
            # Drop existing indexes if they exist
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='{table}'")
            existing_indexes = cursor.fetchall()
            for idx in existing_indexes:
                cursor.execute(f"DROP INDEX IF EXISTS {idx[0]}")
            
            # Add optimized indexes based on compression
            for idx in schema_info[table]["indexes"]:
                cursor.execute(f"CREATE INDEX idx_{table}_{idx} ON {table}({idx})")
        
        # Copy data from original database
        for table in tables:
            orig_cursor.execute(f"SELECT * FROM {table}")
            data = orig_cursor.fetchall()
            if data:
                placeholders = ','.join(['?' for _ in range(len(data[0]))])
                cursor.executemany(
                    f"INSERT INTO {table} VALUES ({placeholders})",
                    data
                )
        
        # Save compression metrics
        self.metrics["database_info"] = {
            "tables": tables,
            "schema": schema_info
        }
        self.metrics["compression_metrics"] = {
            "fidelity": compression_results["fidelity"],
            "parameters": compression_results["parameters"].tolist()  # Convert numpy array to list for JSON
        }
        
        # Save storage metrics
        orig_size = Path(self.original_db_path).stat().st_size
        new_size = Path(self.reconstructed_db_path).stat().st_size
        self.metrics["storage_metrics"] = {
            "original_size": orig_size,
            "optimized_size": new_size,
            "compression_ratio": orig_size / new_size
        }
        
        # Save metrics to file
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        conn.commit()
        conn.close()
        orig_conn.close()
    
    def run_comparison_queries(self) -> None:
        """Run and compare queries on both original and reconstructed databases."""
        test_queries = [
            """
            SELECT c.CustomerName, COUNT(o.OrderID) as OrderCount
            FROM Customers c
            JOIN Orders o ON c.CustomerID = o.CustomerID
            GROUP BY c.CustomerID
            ORDER BY OrderCount DESC
            LIMIT 5
            """,
            """
            SELECT p.ProductName, 
                   COUNT(od.OrderID) as TimesSold,
                   SUM(od.Quantity) as TotalQuantity
            FROM Products p
            JOIN OrderDetails od ON p.ProductID = od.ProductID
            GROUP BY p.ProductID
            HAVING TimesSold > 5
            ORDER BY TotalQuantity DESC
            """,
            """
            SELECT c.CategoryName,
                   COUNT(DISTINCT p.ProductID) as ProductCount,
                   AVG(p.Price) as AvgPrice
            FROM Categories c
            JOIN Products p ON c.CategoryID = p.CategoryID
            GROUP BY c.CategoryID
            ORDER BY ProductCount DESC
            """
        ]
        
        query_results = {}
        for i, query in enumerate(test_queries):
            # Time original database
            orig_conn = sqlite3.connect(self.original_db_path)
            start_time = datetime.now()
            orig_cursor = orig_conn.cursor()
            orig_cursor.execute(query)
            orig_results = orig_cursor.fetchall()
            orig_time = (datetime.now() - start_time).total_seconds()
            
            # Time optimized database
            opt_conn = sqlite3.connect(self.reconstructed_db_path)
            start_time = datetime.now()
            opt_cursor = opt_conn.cursor()
            opt_cursor.execute(query)
            opt_results = opt_cursor.fetchall()
            opt_time = (datetime.now() - start_time).total_seconds()
            
            query_results[f"query_{i+1}"] = {
                "original_time": orig_time,
                "optimized_time": opt_time,
                "speedup": orig_time / opt_time if opt_time > 0 else float('inf'),
                "results_match": orig_results == opt_results
            }
            
            orig_conn.close()
            opt_conn.close()
        
        self.metrics["query_performance"] = query_results
        
        # Save updated metrics
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_metrics(self, output_dir: str = "visualizations") -> None:
        """Generate visualization of compression and performance metrics."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Plot query performance comparison
        query_times = self.metrics["query_performance"]
        queries = list(query_times.keys())
        orig_times = [q["original_time"] for q in query_times.values()]
        opt_times = [q["optimized_time"] for q in query_times.values()]
        
        plt.figure(figsize=(10, 5))
        x = np.arange(len(queries))
        width = 0.35
        plt.bar(x - width/2, orig_times, width, label='Original')
        plt.bar(x + width/2, opt_times, width, label='Optimized')
        plt.title("Query Performance Comparison")
        plt.xlabel("Query")
        plt.ylabel("Execution Time (s)")
        plt.xticks(x, queries)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/query_performance.png")
        plt.close()
        
        # Plot compression metrics
        plt.figure(figsize=(10, 5))
        plt.bar(['Fidelity', 'Compression Ratio'], 
                [self.metrics["compression_metrics"]["fidelity"],
                 self.metrics["storage_metrics"]["compression_ratio"]])
        plt.title("Compression Results")
        plt.ylabel("Value")
        plt.grid(True)
        plt.savefig(f"{output_dir}/compression_metrics.png")
        plt.close()

def main():
    """Run the complete database compression workflow."""
    # Setup paths
    original_db = "tests/database_optimization/ExampleDB/northwind.db"
    compressed_db = "tests/database_optimization/ExampleDB/northwind_compressed.db"
    reconstructed_db = "tests/database_optimization/ExampleDB/northwind_reconstructed.db"
    metrics_file = "tests/database_optimization/ExampleDB/compression_metrics.json"
    
    # Initialize workflow
    workflow = DatabaseCompressionWorkflow(
        original_db,
        compressed_db,
        reconstructed_db,
        metrics_file
    )
    
    # Extract schema information
    print("Extracting schema information...")
    tables, schema_info = workflow.extract_schema_info()
    
    # Prepare quantum features
    print("Preparing quantum features...")
    features = workflow.prepare_quantum_features(schema_info)
    
    # Compress database
    print("Compressing database...")
    compression_results = workflow.compress_database(features, n_latent=3)
    
    # Reconstruct database
    print("Reconstructing optimized database...")
    workflow.reconstruct_database(tables, schema_info, compression_results)
    
    # Run comparison queries
    print("Running performance comparison queries...")
    workflow.run_comparison_queries()
    
    # Generate visualizations
    print("Generating metric visualizations...")
    workflow.plot_metrics()
    
    print("\nWorkflow completed! Results saved to:")
    print(f"- Reconstructed database: {reconstructed_db}")
    print(f"- Compression metrics: {metrics_file}")
    print("- Visualizations: visualizations/")

if __name__ == "__main__":
    main() 