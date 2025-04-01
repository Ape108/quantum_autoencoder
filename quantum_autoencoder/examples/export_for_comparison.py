"""
Export database in a format suitable for online SQL comparison tools.
"""

import sqlite3
from pathlib import Path
import datetime

def export_for_comparison(db_path: str, schema_path: str, data_path: str):
    """
    Export database in a format suitable for online SQL tools.
    
    Args:
        db_path: Path to the SQLite database
        schema_path: Path to save schema SQL
        data_path: Path to save data SQL
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Export schema
    with open(schema_path, 'w') as f:
        f.write("-- Optimized Northwind Database Schema\n")
        f.write(f"-- Generated on: {datetime.datetime.now()}\n")
        f.write("-- Copy this into the 'Schema Panel' of your SQL tool\n\n")
        
        # Drop existing tables
        f.write("-- Drop existing tables\n")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name DESC")
        tables = cursor.fetchall()
        for (table_name,) in tables:
            f.write(f"DROP TABLE IF EXISTS {table_name};\n")
        f.write("\n")
        
        # Create tables and indexes
        for (table_name,) in sorted(tables):  # Sort tables to ensure proper creation order
            # Get CREATE statement
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            create_stmt = cursor.fetchone()[0]
            f.write(f"{create_stmt};\n\n")
            
            # Get indexes
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}' AND sql IS NOT NULL")
            indexes = cursor.fetchall()
            if indexes:
                for (idx_stmt,) in indexes:
                    f.write(f"{idx_stmt};\n")
                f.write("\n")
    
    # Export data
    with open(data_path, 'w') as f:
        f.write("-- Optimized Northwind Database Data\n")
        f.write(f"-- Generated on: {datetime.datetime.now()}\n")
        f.write("-- Copy this into the 'Data / Query Panel' of your SQL tool\n\n")
        
        # Export data for each table
        for (table_name,) in sorted(tables):
            f.write(f"-- Data for {table_name}\n")
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            if rows:
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
        
        # Add example queries to test optimization
        f.write("\n-- Example queries to test optimization\n")
        f.write("-- These queries will use the optimized indexes\n\n")
        
        f.write("-- Query 1: Customer order history (uses idx_orders_customer_date)\n")
        f.write("""SELECT c.CustomerName, o.OrderDate, p.ProductName, od.Quantity
FROM Customers c
JOIN Orders o ON c.CustomerID = o.CustomerID
JOIN OrderDetails od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
WHERE c.Country = 'Germany'
ORDER BY o.OrderDate;\n\n""")
        
        f.write("-- Query 2: Product category analysis (uses idx_products_category)\n")
        f.write("""SELECT c.CategoryName, COUNT(*) as ProductCount, AVG(p.Price) as AvgPrice
FROM Categories c
JOIN Products p ON c.CategoryID = p.CategoryID
GROUP BY c.CategoryName;\n\n""")
        
        f.write("-- Query 3: Order details (uses idx_orderdetails_composite)\n")
        f.write("""SELECT o.OrderID, o.OrderDate, p.ProductName, od.Quantity
FROM Orders o
JOIN OrderDetails od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
WHERE o.OrderDate >= '1996-07-05';\n""")
    
    conn.close()
    print(f"Schema exported to: {schema_path}")
    print(f"Data and queries exported to: {data_path}")
    print("\nInstructions:")
    print("1. Go to https://sqliteonline.com/ or your preferred SQL tool")
    print("2. Copy contents of schema.sql into the editor and run it")
    print("3. Copy contents of data.sql and run it")
    print("4. Try the example queries to test the optimizations")

def main():
    """Export the database for online comparison."""
    # Set paths
    script_dir = Path(__file__).parent.parent.parent
    db_path = script_dir / "tests/database_optimization/ExampleDB/northwind_optimized.db"
    schema_path = script_dir / "tests/database_optimization/ExampleDB/northwind_schema.sql"
    data_path = script_dir / "tests/database_optimization/ExampleDB/northwind_data.sql"
    
    # Export files
    export_for_comparison(str(db_path), str(schema_path), str(data_path))

if __name__ == "__main__":
    main() 