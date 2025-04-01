-- Optimized Northwind Database Schema
-- Generated on: 2025-03-31 18:54:24.246232
-- Copy this into the 'Schema Panel' of your SQL tool

-- Drop existing tables
DROP TABLE IF EXISTS Suppliers;
DROP TABLE IF EXISTS Shippers;
DROP TABLE IF EXISTS Products;
DROP TABLE IF EXISTS Orders;
DROP TABLE IF EXISTS OrderDetails;
DROP TABLE IF EXISTS Employees;
DROP TABLE IF EXISTS Customers;
DROP TABLE IF EXISTS Categories;

CREATE TABLE Categories (
    CategoryID INTEGER PRIMARY KEY,
    CategoryName TEXT,
    Description TEXT
);

CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY,
    CustomerName TEXT,
    ContactName TEXT,
    Address TEXT,
    City TEXT,
    PostalCode TEXT,
    Country TEXT
);

CREATE TABLE Employees (
    EmployeeID INTEGER PRIMARY KEY,
    LastName TEXT,
    FirstName TEXT,
    BirthDate TEXT,
    Photo TEXT,
    Notes TEXT
);

CREATE TABLE OrderDetails (
    OrderDetailID INTEGER PRIMARY KEY,
    OrderID INTEGER,
    ProductID INTEGER,
    Quantity INTEGER,
    FOREIGN KEY (ProductID) REFERENCES Products (ProductID),
    FOREIGN KEY (OrderID) REFERENCES Orders (OrderID)
);

CREATE INDEX idx_orderdetails_composite 
        ON OrderDetails (OrderID, ProductID)
    ;

CREATE TABLE Orders (
    OrderID INTEGER PRIMARY KEY,
    CustomerID INTEGER,
    EmployeeID INTEGER,
    OrderDate TEXT,
    ShipperID INTEGER,
    FOREIGN KEY (ShipperID) REFERENCES Shippers (ShipperID),
    FOREIGN KEY (CustomerID) REFERENCES Customers (CustomerID),
    FOREIGN KEY (EmployeeID) REFERENCES Employees (EmployeeID)
);

CREATE INDEX idx_Orders_ShipperID ON Orders (ShipperID);
CREATE INDEX idx_Orders_CustomerID ON Orders (CustomerID);
CREATE INDEX idx_Orders_EmployeeID ON Orders (EmployeeID);
CREATE INDEX idx_orders_customer_date 
        ON Orders (CustomerID, OrderDate)
    ;

CREATE TABLE Products (
    ProductID INTEGER PRIMARY KEY,
    ProductName TEXT,
    SupplierID INTEGER,
    CategoryID INTEGER,
    Unit TEXT,
    Price REAL,
    FOREIGN KEY (SupplierID) REFERENCES Suppliers (SupplierID),
    FOREIGN KEY (CategoryID) REFERENCES Categories (CategoryID)
);

CREATE INDEX idx_Products_SupplierID ON Products (SupplierID);
CREATE INDEX idx_Products_CategoryID ON Products (CategoryID);
CREATE INDEX idx_products_category 
        ON Products (CategoryID)
    ;

CREATE TABLE Shippers (
    ShipperID INTEGER PRIMARY KEY,
    ShipperName TEXT,
    Phone TEXT
);

CREATE TABLE Suppliers (
    SupplierID INTEGER PRIMARY KEY,
    SupplierName TEXT,
    ContactName TEXT,
    Address TEXT,
    City TEXT,
    PostalCode TEXT,
    Country TEXT,
    Phone TEXT
);

