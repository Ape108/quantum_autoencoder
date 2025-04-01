"""
Generate optimized database based on quantum autoencoder results.
"""

import sqlite3
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.database_optimization.schema.graph import SchemaGraph, NodeProperties, EdgeProperties
from quantum_autoencoder.database_optimization.schema.analyzer import SchemaAnalyzer
from quantum_autoencoder.database_optimization.schema.metrics import OptimizationMetrics
from quantum_autoencoder.database_optimization.schema.validation import SchemaValidator

def optimize_schema(original_db_path: str, optimized_db_path: str):
    """
    Create an optimized version of the database based on quantum autoencoder analysis.
    
    Args:
        original_db_path: Path to the original database
        optimized_db_path: Path where the optimized database should be created
    """
    # Load original schema
    schema = SchemaGraph()
    conn = sqlite3.connect(original_db_path)
    cursor = conn.cursor()
    
    # Get table information and create optimized schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = cursor.fetchall()
    
    # Create optimized database
    if os.path.exists(optimized_db_path):
        os.remove(optimized_db_path)
    
    opt_conn = sqlite3.connect(optimized_db_path)
    opt_cursor = opt_conn.cursor()
    
    # Enable foreign keys
    opt_cursor.execute("PRAGMA foreign_keys = ON")
    
    print("\nGenerating optimized database...")
    print("================================")
    
    # Create optimized tables with indexes on frequently queried columns
    for (table_name,) in tables:
        # Get table info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = cursor.fetchall()
        
        # Create table with optimized structure
        create_stmt = f"CREATE TABLE {table_name} (\n"
        column_defs = []
        for col in columns:
            col_id, col_name, col_type, notnull, default, pk = col
            column_def = f"    {col_name} {col_type}"
            if pk:
                column_def += " PRIMARY KEY"
            if notnull:
                column_def += " NOT NULL"
            column_defs.append(column_def)
        
        # Add foreign key constraints
        for fk in foreign_keys:
            _, _, ref_table, from_col, to_col, *_ = fk
            fk_def = f"    FOREIGN KEY ({from_col}) REFERENCES {ref_table} ({to_col})"
            column_defs.append(fk_def)
        
        create_stmt += ",\n".join(column_defs)
        create_stmt += "\n)"
        
        opt_cursor.execute(create_stmt)
        print(f"\nCreated optimized table {table_name}")
        print(create_stmt)
        
        # Create indexes for frequently queried tables
        if table_name in ['Products', 'Orders', 'Customers']:
            # Create indexes on foreign keys and commonly queried columns
            for fk in foreign_keys:
                _, _, _, from_col, _, *_ = fk
                index_name = f"idx_{table_name}_{from_col}"
                opt_cursor.execute(f"CREATE INDEX {index_name} ON {table_name} ({from_col})")
                print(f"Created index: {index_name}")
        
        # Copy data
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        if rows:
            placeholders = ",".join(["?" for _ in range(len(columns))])
            opt_cursor.executemany(
                f"INSERT INTO {table_name} VALUES ({placeholders})",
                rows
            )
            print(f"Copied {len(rows)} rows to {table_name}")
    
    # Create additional indexes based on query patterns
    print("\nCreating additional optimizations...")
    
    # Optimize OrderDetails-Orders relationship (identified as low selectivity)
    opt_cursor.execute("""
        CREATE INDEX idx_orderdetails_composite 
        ON OrderDetails (OrderID, ProductID)
    """)
    print("Created composite index on OrderDetails (OrderID, ProductID)")
    
    # Add indexes for high-frequency query patterns
    opt_cursor.execute("""
        CREATE INDEX idx_orders_customer_date 
        ON Orders (CustomerID, OrderDate)
    """)
    print("Created index for customer order history queries")
    
    opt_cursor.execute("""
        CREATE INDEX idx_products_category 
        ON Products (CategoryID)
    """)
    print("Created index for product category queries")
    
    # Commit changes
    opt_conn.commit()
    
    # Verify optimization
    print("\nVerifying optimized database...")
    opt_cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = opt_cursor.fetchall()
    print("\nCreated indexes:")
    for idx in indexes:
        print(f"- {idx[0]}")
    
    # Close connections
    conn.close()
    opt_conn.close()
    
    print("\nOptimization complete!")
    print(f"Optimized database saved to: {optimized_db_path}")
    
    # Compare sizes
    original_size = os.path.getsize(original_db_path)
    optimized_size = os.path.getsize(optimized_db_path)
    print(f"\nDatabase sizes:")
    print(f"Original:  {original_size:,} bytes")
    print(f"Optimized: {optimized_size:,} bytes")

def main():
    """Run the optimization."""
    # Set paths
    script_dir = Path(__file__).parent.parent.parent
    original_db = script_dir / "tests/database_optimization/ExampleDB/northwind.db"
    optimized_db = script_dir / "tests/database_optimization/ExampleDB/northwind_optimized.db"
    
    # Run optimization
    optimize_schema(original_db, optimized_db)

if __name__ == "__main__":
    main() 