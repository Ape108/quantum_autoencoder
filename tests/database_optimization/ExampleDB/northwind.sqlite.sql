-- SQLite compatible version of Northwind database

DROP TABLE IF EXISTS OrderDetails;
DROP TABLE IF EXISTS Orders;
DROP TABLE IF EXISTS Products;
DROP TABLE IF EXISTS Categories;
DROP TABLE IF EXISTS Customers;
DROP TABLE IF EXISTS Employees;
DROP TABLE IF EXISTS Shippers;
DROP TABLE IF EXISTS Suppliers;

CREATE TABLE Categories
(      
    CategoryID INTEGER PRIMARY KEY,
    CategoryName TEXT,
    Description TEXT
);

CREATE TABLE Customers
(      
    CustomerID INTEGER PRIMARY KEY,
    CustomerName TEXT,
    ContactName TEXT,
    Address TEXT,
    City TEXT,
    PostalCode TEXT,
    Country TEXT
);

CREATE TABLE Employees
(
    EmployeeID INTEGER PRIMARY KEY,
    LastName TEXT,
    FirstName TEXT,
    BirthDate TEXT,
    Photo TEXT,
    Notes TEXT
);

CREATE TABLE Shippers(
    ShipperID INTEGER PRIMARY KEY,
    ShipperName TEXT,
    Phone TEXT
);

CREATE TABLE Suppliers(
    SupplierID INTEGER PRIMARY KEY,
    SupplierName TEXT,
    ContactName TEXT,
    Address TEXT,
    City TEXT,
    PostalCode TEXT,
    Country TEXT,
    Phone TEXT
);

CREATE TABLE Products(
    ProductID INTEGER PRIMARY KEY,
    ProductName TEXT,
    SupplierID INTEGER,
    CategoryID INTEGER,
    Unit TEXT,
    Price REAL,
    FOREIGN KEY (CategoryID) REFERENCES Categories (CategoryID),
    FOREIGN KEY (SupplierID) REFERENCES Suppliers (SupplierID)
);

CREATE TABLE Orders(
    OrderID INTEGER PRIMARY KEY,
    CustomerID INTEGER,
    EmployeeID INTEGER,
    OrderDate TEXT,
    ShipperID INTEGER,
    FOREIGN KEY (EmployeeID) REFERENCES Employees (EmployeeID),
    FOREIGN KEY (CustomerID) REFERENCES Customers (CustomerID),
    FOREIGN KEY (ShipperID) REFERENCES Shippers (ShipperID)
);

CREATE TABLE OrderDetails(
    OrderDetailID INTEGER PRIMARY KEY,
    OrderID INTEGER,
    ProductID INTEGER,
    Quantity INTEGER,
    FOREIGN KEY (OrderID) REFERENCES Orders (OrderID),
    FOREIGN KEY (ProductID) REFERENCES Products (ProductID)
);

-- Sample data for testing
INSERT INTO Categories VALUES(1,'Beverages','Soft drinks, coffees, teas, beers, and ales');
INSERT INTO Categories VALUES(2,'Condiments','Sweet and savory sauces, relishes, spreads, and seasonings');
INSERT INTO Categories VALUES(3,'Confections','Desserts, candies, and sweet breads');
INSERT INTO Categories VALUES(4,'Dairy Products','Cheeses');
INSERT INTO Categories VALUES(5,'Grains/Cereals','Breads, crackers, pasta, and cereal');

INSERT INTO Customers VALUES(1,'Alfreds Futterkiste','Maria Anders','Obere Str. 57','Berlin','12209','Germany');
INSERT INTO Customers VALUES(2,'Ana Trujillo Emparedados y helados','Ana Trujillo','Avda. de la Constitución 2222','México D.F.','5021','Mexico');
INSERT INTO Customers VALUES(3,'Antonio Moreno Taquería','Antonio Moreno','Mataderos 2312','México D.F.','5023','Mexico');
INSERT INTO Customers VALUES(4,'Around the Horn','Thomas Hardy','120 Hanover Sq.','London','WA1 1DP','UK');
INSERT INTO Customers VALUES(5,'Berglunds snabbköp','Christina Berglund','Berguvsvägen 8','Luleå','S-958 22','Sweden');

INSERT INTO Employees VALUES(1,'Davolio','Nancy','1968-12-08','EmpID1.pic','Education includes a BA in psychology from Colorado State University');
INSERT INTO Employees VALUES(2,'Fuller','Andrew','1952-02-19','EmpID2.pic','Andrew received his BTS commercial and a Ph.D. in international marketing');
INSERT INTO Employees VALUES(3,'Leverling','Janet','1963-08-30','EmpID3.pic','Janet has a BS degree in chemistry from Boston College');

INSERT INTO Shippers VALUES(1,'Speedy Express','(503) 555-9831');
INSERT INTO Shippers VALUES(2,'United Package','(503) 555-3199');
INSERT INTO Shippers VALUES(3,'Federal Shipping','(503) 555-9931');

INSERT INTO Suppliers VALUES(1,'Exotic Liquid','Charlotte Cooper','49 Gilbert St.','London','EC1 4SD','UK','(171) 555-2222');
INSERT INTO Suppliers VALUES(2,'New Orleans Cajun Delights','Shelley Burke','P.O. Box 78934','New Orleans','70117','USA','(100) 555-4822');
INSERT INTO Suppliers VALUES(3,'Grandma Kelly''s Homestead','Regina Murphy','707 Oxford Rd.','Ann Arbor','48104','USA','(313) 555-5735');

INSERT INTO Products VALUES(1,'Chai',1,1,'10 boxes x 20 bags',18);
INSERT INTO Products VALUES(2,'Chang',1,1,'24 - 12 oz bottles',19);
INSERT INTO Products VALUES(3,'Aniseed Syrup',1,2,'12 - 550 ml bottles',10);
INSERT INTO Products VALUES(4,'Chef Anton''s Cajun Seasoning',2,2,'48 - 6 oz jars',22);
INSERT INTO Products VALUES(5,'Chef Anton''s Gumbo Mix',2,2,'36 boxes',21.35);

INSERT INTO Orders VALUES(10248,1,1,'1996-07-04',3);
INSERT INTO Orders VALUES(10249,2,2,'1996-07-05',1);
INSERT INTO Orders VALUES(10250,3,3,'1996-07-08',2);
INSERT INTO Orders VALUES(10251,4,1,'1996-07-08',1);
INSERT INTO Orders VALUES(10252,5,2,'1996-07-09',2);

INSERT INTO OrderDetails VALUES(1,10248,1,12);
INSERT INTO OrderDetails VALUES(2,10248,2,10);
INSERT INTO OrderDetails VALUES(3,10248,3,5);
INSERT INTO OrderDetails VALUES(4,10249,4,9);
INSERT INTO OrderDetails VALUES(5,10249,5,40); 