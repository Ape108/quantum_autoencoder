"""Command line interface for quantum database optimization."""

import argparse
import json
import logging
from pathlib import Path
import sys

from quantum_db_optimizer.core.quantum.optimizer import DatabaseOptimizer
from quantum_db_optimizer.core.schema.analyzer import SchemaAnalyzer

logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optimize database structure using quantum computing"
    )
    
    parser.add_argument(
        "db_path",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=4,
        help="Number of qubits"
    )
    parser.add_argument(
        "--n-latent",
        type=int,
        default=2,
        help="Number of latent qubits"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Analyze schema
        logger.info("Analyzing database schema...")
        analyzer = SchemaAnalyzer(args.db_path)
        schema_analysis = analyzer.analyze()
        
        # Run optimization
        logger.info("Running quantum optimization...")
        optimizer = DatabaseOptimizer(
            n_qubits=args.n_qubits,
            n_latent=args.n_latent,
            output_dir=args.output_dir
        )
        optimization_results = optimizer.optimize(args.db_path)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "schema_analysis.json", "w") as f:
            json.dump(schema_analysis, f, indent=2)
            
        with open(output_dir / "optimization_results.json", "w") as f:
            json.dump(optimization_results, f, indent=2)
            
        # Generate SQL exports
        logger.info("Generating SQL exports...")
        _export_to_sql(args.db_path, output_dir / "original.sql")
        _export_to_sql(
            optimization_results["output_path"],
            output_dir / "optimized.sql"
        )
        
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
        
def _export_to_sql(db_path: str, output_path: Path):
    """Export database to SQL file."""
    import sqlite3
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        with open(output_path, "w") as f:
            # Write schema
            f.write("-- Schema\n\n")
            cursor.execute("""
                SELECT sql 
                FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            for row in cursor.fetchall():
                if row[0]:
                    f.write(f"{row[0]};\n\n")
                    
            # Write data
            f.write("\n-- Data\n\n")
            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                f.write(f"-- Table: {table_name}\n")
                
                # Get column names
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Get data
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                
                for row in rows:
                    values = []
                    for val in row:
                        if val is None:
                            values.append("NULL")
                        elif isinstance(val, str):
                            values.append(f"'{val.replace(chr(39), chr(39)+chr(39))}'")
                        else:
                            values.append(str(val))
                            
                    f.write(
                        f"INSERT INTO {table_name} "
                        f"({', '.join(columns)}) VALUES "
                        f"({', '.join(values)});\n"
                    )
                f.write("\n")

if __name__ == "__main__":
    main() 