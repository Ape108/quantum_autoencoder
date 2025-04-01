import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ResultsFormatter:
    """Formats and exports optimization results in various formats."""
    
    def __init__(self, output_dir: str):
        """
        Initialize formatter.
        
        Args:
            output_dir: Directory to store formatted results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def format_results(
        self,
        original_db: str,
        optimized_db: str,
        compression_metrics: Dict[str, Any]
    ) -> None:
        """
        Format and save results.
        
        Args:
            original_db: Path to original database
            optimized_db: Path to optimized database
            compression_metrics: Compression metrics
        """
        # Create results directory structure
        results_dir = self.output_dir
        sql_dir = results_dir / "sql"
        metrics_dir = results_dir / "metrics"
        sql_dir.mkdir(exist_ok=True)
        metrics_dir.mkdir(exist_ok=True)
        
        # Export databases to SQL
        self._export_db_to_sql(original_db, sql_dir / "original_schema.sql", sql_dir / "original_data.sql")
        self._export_db_to_sql(optimized_db, sql_dir / "optimized_schema.sql", sql_dir / "optimized_data.sql")
        
        # Save compression metrics
        self._save_metrics(compression_metrics, metrics_dir / "compression_metrics.json")
        
        # Generate summary report
        self._generate_summary(
            original_db,
            optimized_db,
            compression_metrics,
            results_dir / "SUMMARY.md"
        )
        
    def _export_db_to_sql(
        self,
        db_path: str,
        schema_output: Path,
        data_output: Path
    ) -> None:
        """Export database schema and data to SQL files."""
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get schema
            cursor.execute("""
                SELECT sql 
                FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            schema = cursor.fetchall()
            
            # Write schema
            with open(schema_output, 'w') as f:
                f.write("-- Database Schema\n\n")
                for table_sql in schema:
                    if table_sql[0]:  # Some system tables might have NULL sql
                        f.write(f"{table_sql[0]};\n\n")
            
            # Get table names
            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = cursor.fetchall()
            
            # Write data
            with open(data_output, 'w') as f:
                f.write("-- Database Data\n\n")
                for table in tables:
                    table_name = table[0]
                    f.write(f"-- Table: {table_name}\n")
                    
                    # Get column names
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    # Get data
                    cursor.execute(f"SELECT * FROM {table_name}")
                    rows = cursor.fetchall()
                    
                    # Write INSERT statements
                    for row in rows:
                        values = []
                        for val in row:
                            if val is None:
                                values.append('NULL')
                            elif isinstance(val, str):
                                values.append(f"'{val.replace(chr(39), chr(39)+chr(39))}'")
                            else:
                                values.append(str(val))
                        
                        f.write(
                            f"INSERT INTO {table_name} ({', '.join(columns)}) "
                            f"VALUES ({', '.join(values)});\n"
                        )
                    f.write("\n")
    
    def _save_metrics(self, metrics: Dict[str, Any], output_path: Path) -> None:
        """Save metrics as formatted JSON."""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def _generate_summary(
        self,
        original_db: str,
        optimized_db: str,
        metrics: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Generate a human-readable summary report."""
        with sqlite3.connect(original_db) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            original_tables = cursor.fetchall()
            
        with sqlite3.connect(optimized_db) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            optimized_tables = cursor.fetchall()
            
        with open(output_path, 'w') as f:
            f.write("# Quantum Database Optimization Results\n\n")
            
            # Database Overview
            f.write("## Database Overview\n\n")
            f.write("### Original Database\n")
            f.write(f"- Location: `{original_db}`\n")
            f.write(f"- Tables: {len(original_tables)}\n")
            f.write("  - " + "\n  - ".join([t[0] for t in original_tables]) + "\n\n")
            
            f.write("### Optimized Database\n")
            f.write(f"- Location: `{optimized_db}`\n")
            f.write(f"- Tables: {len(optimized_tables)}\n")
            f.write("  - " + "\n  - ".join([t[0] for t in optimized_tables]) + "\n\n")
            
            # Compression Results
            f.write("## Compression Results\n\n")
            f.write("### Quantum Compression\n")
            f.write(f"- Number of qubits: {metrics.get('n_qubits', 'N/A')}\n")
            f.write(f"- Latent qubits: {metrics.get('n_latent', 'N/A')}\n")
            f.write(f"- Compression ratio: {metrics.get('compression_ratio', 'N/A'):.2f}\n")
            f.write(f"- Final loss: {metrics.get('final_loss', 'N/A'):.4f}\n\n")
            
            # File Locations
            f.write("## Result Files\n\n")
            f.write("### SQL Files\n")
            f.write("- Original database:\n")
            f.write("  - Schema: `sql/original_schema.sql`\n")
            f.write("  - Data: `sql/original_data.sql`\n")
            f.write("- Optimized database:\n")
            f.write("  - Schema: `sql/optimized_schema.sql`\n")
            f.write("  - Data: `sql/optimized_data.sql`\n\n")
            
            f.write("### Metrics and Analysis\n")
            f.write("- Detailed metrics: `metrics/compression_metrics.json`\n")
            if os.path.exists(self.output_dir / "compression_heatmap.png"):
                f.write("- Compression heatmap: `compression_heatmap.png`\n")
            if os.path.exists(self.output_dir / "query_analysis.png"):
                f.write("- Query analysis: `query_analysis.png`\n")
            if os.path.exists(self.output_dir / "schema_relationships.png"):
                f.write("- Schema relationships: `schema_relationships.png`\n") 