-- Optimized Northwind Database Data
-- Generated on: 2025-03-31 18:54:24.246747
-- Copy this into the 'Data / Query Panel' of your SQL tool

-- Data for Categories
INSERT INTO Categories (CategoryID, CategoryName, Description) VALUES (1, 'Beverages', 'Soft drinks, coffees, teas, beers, and ales');
INSERT INTO Categories (CategoryID, CategoryName, Description) VALUES (2, 'Condiments', 'Sweet and savory sauces, relishes, spreads, and seasonings');
INSERT INTO Categories (CategoryID, CategoryName, Description) VALUES (3, 'Confections', 'Desserts, candies, and sweet breads');
INSERT INTO Categories (CategoryID, CategoryName, Description) VALUES (4, 'Dairy Products', 'Cheeses');
INSERT INTO Categories (CategoryID, CategoryName, Description) VALUES (5, 'Grains/Cereals', 'Breads, crackers, pasta, and cereal');

-- Data for Customers
INSERT INTO Customers (CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country) VALUES (1, 'Alfreds Futterkiste', 'Maria Anders', 'Obere Str. 57', 'Berlin', '12209', 'Germany');
INSERT INTO Customers (CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country) VALUES (2, 'Ana Trujillo Emparedados y helados', 'Ana Trujillo', 'Avda. de la Constitución 2222', 'México D.F.', '5021', 'Mexico');
INSERT INTO Customers (CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country) VALUES (3, 'Antonio Moreno Taquería', 'Antonio Moreno', 'Mataderos 2312', 'México D.F.', '5023', 'Mexico');
INSERT INTO Customers (CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country) VALUES (4, 'Around the Horn', 'Thomas Hardy', '120 Hanover Sq.', 'London', 'WA1 1DP', 'UK');
INSERT INTO Customers (CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country) VALUES (5, 'Berglunds snabbköp', 'Christina Berglund', 'Berguvsvägen 8', 'Luleå', 'S-958 22', 'Sweden');

-- Data for Employees
INSERT INTO Employees (EmployeeID, LastName, FirstName, BirthDate, Photo, Notes) VALUES (1, 'Davolio', 'Nancy', '1968-12-08', 'EmpID1.pic', 'Education includes a BA in psychology from Colorado State University');
INSERT INTO Employees (EmployeeID, LastName, FirstName, BirthDate, Photo, Notes) VALUES (2, 'Fuller', 'Andrew', '1952-02-19', 'EmpID2.pic', 'Andrew received his BTS commercial and a Ph.D. in international marketing');
INSERT INTO Employees (EmployeeID, LastName, FirstName, BirthDate, Photo, Notes) VALUES (3, 'Leverling', 'Janet', '1963-08-30', 'EmpID3.pic', 'Janet has a BS degree in chemistry from Boston College');

-- Data for OrderDetails
INSERT INTO OrderDetails (OrderDetailID, OrderID, ProductID, Quantity) VALUES (1, 10248, 1, 12);
INSERT INTO OrderDetails (OrderDetailID, OrderID, ProductID, Quantity) VALUES (2, 10248, 2, 10);
INSERT INTO OrderDetails (OrderDetailID, OrderID, ProductID, Quantity) VALUES (3, 10248, 3, 5);
INSERT INTO OrderDetails (OrderDetailID, OrderID, ProductID, Quantity) VALUES (4, 10249, 4, 9);
INSERT INTO OrderDetails (OrderDetailID, OrderID, ProductID, Quantity) VALUES (5, 10249, 5, 40);

-- Data for Orders
INSERT INTO Orders (OrderID, CustomerID, EmployeeID, OrderDate, ShipperID) VALUES (10248, 1, 1, '1996-07-04', 3);
INSERT INTO Orders (OrderID, CustomerID, EmployeeID, OrderDate, ShipperID) VALUES (10249, 2, 2, '1996-07-05', 1);
INSERT INTO Orders (OrderID, CustomerID, EmployeeID, OrderDate, ShipperID) VALUES (10250, 3, 3, '1996-07-08', 2);
INSERT INTO Orders (OrderID, CustomerID, EmployeeID, OrderDate, ShipperID) VALUES (10251, 4, 1, '1996-07-08', 1);
INSERT INTO Orders (OrderID, CustomerID, EmployeeID, OrderDate, ShipperID) VALUES (10252, 5, 2, '1996-07-09', 2);

-- Data for Products
INSERT INTO Products (ProductID, ProductName, SupplierID, CategoryID, Unit, Price) VALUES (1, 'Chai', 1, 1, '10 boxes x 20 bags', 18.0);
INSERT INTO Products (ProductID, ProductName, SupplierID, CategoryID, Unit, Price) VALUES (2, 'Chang', 1, 1, '24 - 12 oz bottles', 19.0);
INSERT INTO Products (ProductID, ProductName, SupplierID, CategoryID, Unit, Price) VALUES (3, 'Aniseed Syrup', 1, 2, '12 - 550 ml bottles', 10.0);
INSERT INTO Products (ProductID, ProductName, SupplierID, CategoryID, Unit, Price) VALUES (4, 'Chef Anton''s Cajun Seasoning', 2, 2, '48 - 6 oz jars', 22.0);
INSERT INTO Products (ProductID, ProductName, SupplierID, CategoryID, Unit, Price) VALUES (5, 'Chef Anton''s Gumbo Mix', 2, 2, '36 boxes', 21.35);

-- Data for Shippers
INSERT INTO Shippers (ShipperID, ShipperName, Phone) VALUES (1, 'Speedy Express', '(503) 555-9831');
INSERT INTO Shippers (ShipperID, ShipperName, Phone) VALUES (2, 'United Package', '(503) 555-3199');
INSERT INTO Shippers (ShipperID, ShipperName, Phone) VALUES (3, 'Federal Shipping', '(503) 555-9931');

-- Data for Suppliers
INSERT INTO Suppliers (SupplierID, SupplierName, ContactName, Address, City, PostalCode, Country, Phone) VALUES (1, 'Exotic Liquid', 'Charlotte Cooper', '49 Gilbert St.', 'London', 'EC1 4SD', 'UK', '(171) 555-2222');
INSERT INTO Suppliers (SupplierID, SupplierName, ContactName, Address, City, PostalCode, Country, Phone) VALUES (2, 'New Orleans Cajun Delights', 'Shelley Burke', 'P.O. Box 78934', 'New Orleans', '70117', 'USA', '(100) 555-4822');
INSERT INTO Suppliers (SupplierID, SupplierName, ContactName, Address, City, PostalCode, Country, Phone) VALUES (3, 'Grandma Kelly''s Homestead', 'Regina Murphy', '707 Oxford Rd.', 'Ann Arbor', '48104', 'USA', '(313) 555-5735');


-- Example queries to test optimization
-- These queries will use the optimized indexes

-- Query 1: Customer order history (uses idx_orders_customer_date)
SELECT c.CustomerName, o.OrderDate, p.ProductName, od.Quantity
FROM Customers c
JOIN Orders o ON c.CustomerID = o.CustomerID
JOIN OrderDetails od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
WHERE c.Country = 'Germany'
ORDER BY o.OrderDate;

-- Query 2: Product category analysis (uses idx_products_category)
SELECT c.CategoryName, COUNT(*) as ProductCount, AVG(p.Price) as AvgPrice
FROM Categories c
JOIN Products p ON c.CategoryID = p.CategoryID
GROUP BY c.CategoryName;

-- Query 3: Order details (uses idx_orderdetails_composite)
SELECT o.OrderID, o.OrderDate, p.ProductName, od.Quantity
FROM Orders o
JOIN OrderDetails od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
WHERE o.OrderDate >= '1996-07-05';
