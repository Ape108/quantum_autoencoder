"""Create a test database for optimization examples."""

import sqlite3
from pathlib import Path

def create_test_db(output_path: str):
    """Create test database."""
    with sqlite3.connect(output_path) as conn:
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE Categories (
                CategoryID INTEGER PRIMARY KEY,
                CategoryName TEXT,
                Description TEXT
            );
        """)
        
        cursor.execute("""
            CREATE TABLE Products (
                ProductID INTEGER PRIMARY KEY,
                ProductName TEXT,
                CategoryID INTEGER,
                UnitPrice REAL,
                FOREIGN KEY (CategoryID) REFERENCES Categories (CategoryID)
            );
        """)
        
        cursor.execute("""
            CREATE TABLE Orders (
                OrderID INTEGER PRIMARY KEY,
                OrderDate TEXT,
                CustomerName TEXT
            );
        """)
        
        cursor.execute("""
            CREATE TABLE OrderDetails (
                OrderDetailID INTEGER PRIMARY KEY,
                OrderID INTEGER,
                ProductID INTEGER,
                Quantity INTEGER,
                FOREIGN KEY (OrderID) REFERENCES Orders (OrderID),
                FOREIGN KEY (ProductID) REFERENCES Products (ProductID)
            );
        """)
        
        # Insert sample data
        cursor.executemany(
            "INSERT INTO Categories VALUES (?, ?, ?)",
            [
                (1, "Beverages", "Soft drinks, coffees, teas"),
                (2, "Condiments", "Sweet and savory sauces"),
                (3, "Confections", "Desserts, candies, and breads")
            ]
        )
        
        cursor.executemany(
            "INSERT INTO Products VALUES (?, ?, ?, ?)",
            [
                (1, "Chai", 1, 18.00),
                (2, "Chang", 1, 19.00),
                (3, "Aniseed Syrup", 2, 10.00),
                (4, "Chef Anton's Cajun Seasoning", 2, 22.00),
                (5, "Chocolade", 3, 12.75)
            ]
        )
        
        cursor.executemany(
            "INSERT INTO Orders VALUES (?, ?, ?)",
            [
                (1, "2024-03-31", "John Smith"),
                (2, "2024-03-31", "Jane Doe"),
                (3, "2024-03-31", "Bob Wilson")
            ]
        )
        
        cursor.executemany(
            "INSERT INTO OrderDetails VALUES (?, ?, ?, ?)",
            [
                (1, 1, 1, 2),  # 2 Chai
                (2, 1, 3, 1),  # 1 Aniseed Syrup
                (3, 2, 5, 3),  # 3 Chocolade
                (4, 3, 2, 1),  # 1 Chang
                (5, 3, 4, 2)   # 2 Cajun Seasoning
            ]
        )
        
        conn.commit()

def main():
    """Create test database in examples directory."""
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "test.db"
    create_test_db(str(output_path))
    print(f"Test database created at: {output_path}")

if __name__ == "__main__":
    main() 