-- Performance comparison queries for Northwind database
-- Run these queries on both original and optimized databases to see the difference

-- First, enable query planning output
.timer on
.explain on

-- Test 1: OrderDetails-Orders relationship optimization
-- This uses the composite index in the optimized version
EXPLAIN QUERY PLAN
SELECT o.OrderID, o.OrderDate, p.ProductName, od.Quantity
FROM Orders o
JOIN OrderDetails od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
WHERE o.OrderDate BETWEEN '1996-07-04' AND '1996-07-08'
ORDER BY o.OrderDate;

-- Test 2: Customer order history with date range
-- Uses the optimized customer_order_date index
EXPLAIN QUERY PLAN
SELECT c.CustomerName, 
       COUNT(o.OrderID) as OrderCount,
       GROUP_CONCAT(p.ProductName) as Products
FROM Customers c
JOIN Orders o ON c.CustomerID = o.CustomerID
JOIN OrderDetails od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
WHERE o.OrderDate >= '1996-07-05'
GROUP BY c.CustomerID
ORDER BY OrderCount DESC;

-- Test 3: Product category query with aggregation
-- Uses the optimized product category index
EXPLAIN QUERY PLAN
SELECT c.CategoryName,
       COUNT(DISTINCT p.ProductID) as ProductCount,
       AVG(od.Quantity) as AvgOrderQuantity,
       SUM(p.Price * od.Quantity) as TotalRevenue
FROM Categories c
JOIN Products p ON c.CategoryID = p.CategoryID
JOIN OrderDetails od ON p.ProductID = od.ProductID
GROUP BY c.CategoryID
HAVING COUNT(DISTINCT p.ProductID) > 0
ORDER BY TotalRevenue DESC;

-- Test 4: Complex join with multiple conditions
-- Tests overall index optimization
EXPLAIN QUERY PLAN
SELECT 
    c.CustomerName,
    p.ProductName,
    cat.CategoryName,
    o.OrderDate,
    od.Quantity,
    s.SupplierName,
    e.FirstName || ' ' || e.LastName as EmployeeName
FROM Orders o
JOIN Customers c ON o.CustomerID = c.CustomerID
JOIN Employees e ON o.EmployeeID = e.EmployeeID
JOIN OrderDetails od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
JOIN Categories cat ON p.CategoryID = cat.CategoryID
JOIN Suppliers s ON p.SupplierID = s.SupplierID
WHERE c.Country IN ('Germany', 'UK')
  AND o.OrderDate >= '1996-07-05'
  AND p.Price > 15
ORDER BY o.OrderDate, p.ProductName;

-- Test 5: Subquery optimization test
-- Tests index usage in subqueries
EXPLAIN QUERY PLAN
SELECT 
    c.CategoryName,
    p.ProductName,
    p.Price,
    (SELECT COUNT(*) 
     FROM OrderDetails od 
     WHERE od.ProductID = p.ProductID) as TimesOrdered,
    (SELECT AVG(od2.Quantity)
     FROM OrderDetails od2
     WHERE od2.ProductID = p.ProductID) as AvgQuantity
FROM Products p
JOIN Categories c ON p.CategoryID = c.CategoryID
WHERE EXISTS (
    SELECT 1 
    FROM OrderDetails od3
    JOIN Orders o ON od3.OrderID = o.OrderID
    WHERE od3.ProductID = p.ProductID
    AND o.OrderDate >= '1996-07-05'
)
ORDER BY TimesOrdered DESC;

-- To compare execution times, run each query without EXPLAIN:
-- 1. First in the original database
-- 2. Then in the optimized database
-- Look for differences in:
-- - Query plan (number of table scans vs index usage)
-- - Execution time
-- - Number of steps in the query plan 