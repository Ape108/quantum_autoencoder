"""
Create a test database for quantum optimization testing.
"""

import sqlite3
from pathlib import Path

def create_test_database(db_path: str):
    """Create a test database with sample tables and data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE Categories (
            CategoryID INTEGER PRIMARY KEY,
            CategoryName TEXT,
            Description TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE Products (
            ProductID INTEGER PRIMARY KEY,
            ProductName TEXT,
            CategoryID INTEGER,
            UnitPrice REAL,
            UnitsInStock INTEGER,
            FOREIGN KEY (CategoryID) REFERENCES Categories(CategoryID)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE Orders (
            OrderID INTEGER PRIMARY KEY,
            OrderDate TEXT,
            TotalAmount REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE OrderDetails (
            OrderDetailID INTEGER PRIMARY KEY,
            OrderID INTEGER,
            ProductID INTEGER,
            Quantity INTEGER,
            UnitPrice REAL,
            FOREIGN KEY (OrderID) REFERENCES Orders(OrderID),
            FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
        )
    """)
    
    # Insert sample data
    cursor.executemany(
        "INSERT INTO Categories (CategoryName, Description) VALUES (?, ?)",
        [
            ("Beverages", "Soft drinks, coffees, teas, beers, and ales"),
            ("Condiments", "Sweet and savory sauces, relishes, spreads, and seasonings"),
            ("Confections", "Desserts, candies, and sweet breads"),
            ("Dairy Products", "Cheeses"),
            ("Grains/Cereals", "Breads, crackers, pasta, and cereal")
        ]
    )
    
    cursor.executemany(
        "INSERT INTO Products (ProductName, CategoryID, UnitPrice, UnitsInStock) VALUES (?, ?, ?, ?)",
        [
            ("Chai", 1, 18.00, 39),
            ("Chang", 1, 19.00, 17),
            ("Aniseed Syrup", 2, 10.00, 13),
            ("Chef Anton's Cajun Seasoning", 2, 22.00, 53),
            ("Chef Anton's Gumbo Mix", 2, 21.35, 0)
        ]
    )
    
    cursor.executemany(
        "INSERT INTO Orders (OrderDate, TotalAmount) VALUES (?, ?)",
        [
            ("2024-03-31", 428.00),
            ("2024-03-31", 1842.00),
            ("2024-03-31", 440.00),
            ("2024-03-31", 2310.00),
            ("2024-03-31", 580.00)
        ]
    )
    
    cursor.executemany(
        "INSERT INTO OrderDetails (OrderID, ProductID, Quantity, UnitPrice) VALUES (?, ?, ?, ?)",
        [
            (1, 1, 12, 18.00),
            (1, 3, 10, 10.00),
            (2, 2, 24, 19.00),
            (2, 4, 12, 22.00),
            (3, 1, 20, 18.00),
            (3, 3, 8, 10.00)
        ]
    )
    
    conn.commit()
    conn.close()

def main():
    """Create test database."""
    db_path = Path("database.db")
    create_test_database(str(db_path))
    print(f"Test database created at: {db_path}")

if __name__ == "__main__":
    main() 