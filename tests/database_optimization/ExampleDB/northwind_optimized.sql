-- Optimized Northwind Database
-- Generated on: 2025-03-31 18:52:17.903500
-- This file contains the complete optimized database structure and data


-- Table: Categories
CREATE TABLE Categories (
    CategoryID INTEGER PRIMARY KEY,
    CategoryName TEXT,
    Description TEXT
);

-- Data for table Categories
INSERT INTO Categories (CategoryID, CategoryName, Description) VALUES (1, 'Beverages', 'Soft drinks, coffees, teas, beers, and ales');
INSERT INTO Categories (CategoryID, CategoryName, Description) VALUES (2, 'Condiments', 'Sweet and savory sauces, relishes, spreads, and seasonings');
INSERT INTO Categories (CategoryID, CategoryName, Description) VALUES (3, 'Confections', 'Desserts, candies, and sweet breads');
INSERT INTO Categories (CategoryID, CategoryName, Description) VALUES (4, 'Dairy Products', 'Cheeses');
INSERT INTO Categories (CategoryID, CategoryName, Description) VALUES (5, 'Grains/Cereals', 'Breads, crackers, pasta, and cereal');


-- Table: Customers
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY,
    CustomerName TEXT,
    ContactName TEXT,
    Address TEXT,
    City TEXT,
    PostalCode TEXT,
    Country TEXT
);

-- Data for table Customers
INSERT INTO Customers (CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country) VALUES (1, 'Alfreds Futterkiste', 'Maria Anders', 'Obere Str. 57', 'Berlin', '12209', 'Germany');
INSERT INTO Customers (CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country) VALUES (2, 'Ana Trujillo Emparedados y helados', 'Ana Trujillo', 'Avda. de la Constitución 2222', 'México D.F.', '5021', 'Mexico');
INSERT INTO Customers (CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country) VALUES (3, 'Antonio Moreno Taquería', 'Antonio Moreno', 'Mataderos 2312', 'México D.F.', '5023', 'Mexico');
INSERT INTO Customers (CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country) VALUES (4, 'Around the Horn', 'Thomas Hardy', '120 Hanover Sq.', 'London', 'WA1 1DP', 'UK');
INSERT INTO Customers (CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country) VALUES (5, 'Berglunds snabbköp', 'Christina Berglund', 'Berguvsvägen 8', 'Luleå', 'S-958 22', 'Sweden');


-- Table: Employees
CREATE TABLE Employees (
    EmployeeID INTEGER PRIMARY KEY,
    LastName TEXT,
    FirstName TEXT,
    BirthDate TEXT,
    Photo TEXT,
    Notes TEXT
);

-- Data for table Employees
INSERT INTO Employees (EmployeeID, LastName, FirstName, BirthDate, Photo, Notes) VALUES (1, 'Davolio', 'Nancy', '1968-12-08', 'EmpID1.pic', 'Education includes a BA in psychology from Colorado State University');
INSERT INTO Employees (EmployeeID, LastName, FirstName, BirthDate, Photo, Notes) VALUES (2, 'Fuller', 'Andrew', '1952-02-19', 'EmpID2.pic', 'Andrew received his BTS commercial and a Ph.D. in international marketing');
INSERT INTO Employees (EmployeeID, LastName, FirstName, BirthDate, Photo, Notes) VALUES (3, 'Leverling', 'Janet', '1963-08-30', 'EmpID3.pic', 'Janet has a BS degree in chemistry from Boston College');


-- Table: Shippers
CREATE TABLE Shippers (
    ShipperID INTEGER PRIMARY KEY,
    ShipperName TEXT,
    Phone TEXT
);

-- Data for table Shippers
INSERT INTO Shippers (ShipperID, ShipperName, Phone) VALUES (1, 'Speedy Express', '(503) 555-9831');
INSERT INTO Shippers (ShipperID, ShipperName, Phone) VALUES (2, 'United Package', '(503) 555-3199');
INSERT INTO Shippers (ShipperID, ShipperName, Phone) VALUES (3, 'Federal Shipping', '(503) 555-9931');


-- Table: Suppliers
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

-- Data for table Suppliers
INSERT INTO Suppliers (SupplierID, SupplierName, ContactName, Address, City, PostalCode, Country, Phone) VALUES (1, 'Exotic Liquid', 'Charlotte Cooper', '49 Gilbert St.', 'London', 'EC1 4SD', 'UK', '(171) 555-2222');
INSERT INTO Suppliers (SupplierID, SupplierName, ContactName, Address, City, PostalCode, Country, Phone) VALUES (2, 'New Orleans Cajun Delights', 'Shelley Burke', 'P.O. Box 78934', 'New Orleans', '70117', 'USA', '(100) 555-4822');
INSERT INTO Suppliers (SupplierID, SupplierName, ContactName, Address, City, PostalCode, Country, Phone) VALUES (3, 'Grandma Kelly''s Homestead', 'Regina Murphy', '707 Oxford Rd.', 'Ann Arbor', '48104', 'USA', '(313) 555-5735');


-- Table: Products
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

-- Indexes for table Products
CREATE INDEX idx_Products_SupplierID ON Products (SupplierID);
CREATE INDEX idx_Products_CategoryID ON Products (CategoryID);
CREATE INDEX idx_products_category 
        ON Products (CategoryID)
    ;

-- Data for table Products
INSERT INTO Products (ProductID, ProductName, SupplierID, CategoryID, Unit, Price) VALUES (1, 'Chai', 1, 1, '10 boxes x 20 bags', 18.0);
INSERT INTO Products (ProductID, ProductName, SupplierID, CategoryID, Unit, Price) VALUES (2, 'Chang', 1, 1, '24 - 12 oz bottles', 19.0);
INSERT INTO Products (ProductID, ProductName, SupplierID, CategoryID, Unit, Price) VALUES (3, 'Aniseed Syrup', 1, 2, '12 - 550 ml bottles', 10.0);
INSERT INTO Products (ProductID, ProductName, SupplierID, CategoryID, Unit, Price) VALUES (4, 'Chef Anton''s Cajun Seasoning', 2, 2, '48 - 6 oz jars', 22.0);
INSERT INTO Products (ProductID, ProductName, SupplierID, CategoryID, Unit, Price) VALUES (5, 'Chef Anton''s Gumbo Mix', 2, 2, '36 boxes', 21.35);


-- Table: Orders
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

-- Indexes for table Orders
CREATE INDEX idx_Orders_ShipperID ON Orders (ShipperID);
CREATE INDEX idx_Orders_CustomerID ON Orders (CustomerID);
CREATE INDEX idx_Orders_EmployeeID ON Orders (EmployeeID);
CREATE INDEX idx_orders_customer_date 
        ON Orders (CustomerID, OrderDate)
    ;

-- Data for table Orders
INSERT INTO Orders (OrderID, CustomerID, EmployeeID, OrderDate, ShipperID) VALUES (10248, 1, 1, '1996-07-04', 3);
INSERT INTO Orders (OrderID, CustomerID, EmployeeID, OrderDate, ShipperID) VALUES (10249, 2, 2, '1996-07-05', 1);
INSERT INTO Orders (OrderID, CustomerID, EmployeeID, OrderDate, ShipperID) VALUES (10250, 3, 3, '1996-07-08', 2);
INSERT INTO Orders (OrderID, CustomerID, EmployeeID, OrderDate, ShipperID) VALUES (10251, 4, 1, '1996-07-08', 1);
INSERT INTO Orders (OrderID, CustomerID, EmployeeID, OrderDate, ShipperID) VALUES (10252, 5, 2, '1996-07-09', 2);


-- Table: OrderDetails
CREATE TABLE OrderDetails (
    OrderDetailID INTEGER PRIMARY KEY,
    OrderID INTEGER,
    ProductID INTEGER,
    Quantity INTEGER,
    FOREIGN KEY (ProductID) REFERENCES Products (ProductID),
    FOREIGN KEY (OrderID) REFERENCES Orders (OrderID)
);

-- Indexes for table OrderDetails
CREATE INDEX idx_orderdetails_composite 
        ON OrderDetails (OrderID, ProductID)
    ;

-- Data for table OrderDetails
INSERT INTO OrderDetails (OrderDetailID, OrderID, ProductID, Quantity) VALUES (1, 10248, 1, 12);
INSERT INTO OrderDetails (OrderDetailID, OrderID, ProductID, Quantity) VALUES (2, 10248, 2, 10);
INSERT INTO OrderDetails (OrderDetailID, OrderID, ProductID, Quantity) VALUES (3, 10248, 3, 5);
INSERT INTO OrderDetails (OrderDetailID, OrderID, ProductID, Quantity) VALUES (4, 10249, 4, 9);
INSERT INTO OrderDetails (OrderDetailID, OrderID, ProductID, Quantity) VALUES (5, 10249, 5, 40);


-- Database Statistics
-- ===================

-- Table Row Counts:
-- Categories: 5 rows
-- Customers: 5 rows
-- Employees: 3 rows
-- Shippers: 3 rows
-- Suppliers: 3 rows
-- Products: 5 rows
-- Orders: 5 rows
-- OrderDetails: 5 rows

-- Indexes:
-- idx_Products_SupplierID on table Products
-- idx_Products_CategoryID on table Products
-- idx_Orders_ShipperID on table Orders
-- idx_Orders_CustomerID on table Orders
-- idx_Orders_EmployeeID on table Orders
-- idx_orderdetails_composite on table OrderDetails
-- idx_orders_customer_date on table Orders
-- idx_products_category on table Products

-- Optimization Notes:
-- 1. Added composite index on OrderDetails (OrderID, ProductID) for better join performance
-- 2. Created customer order history index (CustomerID, OrderDate) for faster order lookups
-- 3. Added product category index for improved category-based queries
-- 4. Optimized foreign key relationships with dedicated indexes
-- 5. Maintained all original data integrity constraints
