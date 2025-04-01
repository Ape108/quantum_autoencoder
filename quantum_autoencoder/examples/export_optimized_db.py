"""
Export optimized database to readable SQL format.
"""

import sqlite3
from pathlib import Path
import datetime

def export_database_to_sql(db_path: str, output_path: str):
    """
    Export SQLite database to readable SQL file.
    
    Args:
        db_path: Path to the SQLite database
        output_path: Path where to save the SQL file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("-- Optimized Northwind Database\n")
        f.write(f"-- Generated on: {datetime.datetime.now()}\n")
        f.write("-- This file contains the complete optimized database structure and data\n\n")
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall()
        
        # Export each table structure and data
        for (table_name,) in tables:
            f.write(f"\n-- Table: {table_name}\n")
            
            # Get CREATE statement
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            create_stmt = cursor.fetchone()[0]
            f.write(f"{create_stmt};\n\n")
            
            # Get indexes
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}' AND sql IS NOT NULL")
            indexes = cursor.fetchall()
            if indexes:
                f.write(f"-- Indexes for table {table_name}\n")
                for (idx_stmt,) in indexes:
                    f.write(f"{idx_stmt};\n")
                f.write("\n")
            
            # Get data
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            if rows:
                f.write(f"-- Data for table {table_name}\n")
                # Get column names
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                # Write INSERT statements
                for row in rows:
                    values = []
                    for val in row:
                        if val is None:
                            values.append("NULL")
                        elif isinstance(val, str):
                            values.append(f"'{val.replace(chr(39), chr(39)+chr(39))}'")
                        else:
                            values.append(str(val))
                    f.write(f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({', '.join(values)});\n")
                f.write("\n")
        
        # Export database statistics
        f.write("\n-- Database Statistics\n")
        f.write("-- ===================\n")
        
        # Table statistics
        f.write("\n-- Table Row Counts:\n")
        for (table_name,) in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            f.write(f"-- {table_name}: {count} rows\n")
        
        # Index statistics
        f.write("\n-- Indexes:\n")
        cursor.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")
        indexes = cursor.fetchall()
        for index_name, table_name in indexes:
            f.write(f"-- {index_name} on table {table_name}\n")
        
        # Write optimization notes
        f.write("\n-- Optimization Notes:\n")
        f.write("-- 1. Added composite index on OrderDetails (OrderID, ProductID) for better join performance\n")
        f.write("-- 2. Created customer order history index (CustomerID, OrderDate) for faster order lookups\n")
        f.write("-- 3. Added product category index for improved category-based queries\n")
        f.write("-- 4. Optimized foreign key relationships with dedicated indexes\n")
        f.write("-- 5. Maintained all original data integrity constraints\n")
    
    conn.close()
    print(f"Database exported successfully to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size:,} bytes")

def main():
    """Export the optimized database to SQL."""
    # Set paths
    script_dir = Path(__file__).parent.parent.parent
    db_path = script_dir / "tests/database_optimization/ExampleDB/northwind_optimized.db"
    sql_path = script_dir / "tests/database_optimization/ExampleDB/northwind_optimized.sql"
    
    # Export database
    export_database_to_sql(str(db_path), str(sql_path))
    
    # Print first few lines of the exported file
    print("\nPreview of exported SQL file:")
    print("=============================")
    with open(sql_path, 'r') as f:
        print(''.join(f.readlines()[:20]))

if __name__ == "__main__":
    main() 