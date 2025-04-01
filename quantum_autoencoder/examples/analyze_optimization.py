"""
Analyze optimization differences between original and reconstructed databases.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

class OptimizationAnalyzer:
    def __init__(self, original_db: str, reconstructed_db: str):
        """Initialize database connections."""
        self.orig_conn = sqlite3.connect(original_db)
        self.recon_conn = sqlite3.connect(reconstructed_db)
        self.orig_cursor = self.orig_conn.cursor()
        self.recon_cursor = self.recon_conn.cursor()
        
        # Enable foreign keys
        self.orig_cursor.execute("PRAGMA foreign_keys = ON")
        self.recon_cursor.execute("PRAGMA foreign_keys = ON")
    
    def get_table_structure(self, cursor: sqlite3.Cursor, table: str) -> Dict:
        """Get detailed table structure including indexes and constraints."""
        structure = {}
        
        # Get CREATE TABLE statement
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
        structure['create_stmt'] = cursor.fetchone()[0]
        
        # Get indexes
        cursor.execute(f"SELECT * FROM sqlite_master WHERE type='index' AND tbl_name='{table}'")
        structure['indexes'] = cursor.fetchall()
        
        # Get detailed index info
        structure['index_info'] = {}
        for idx in structure['indexes']:
            cursor.execute(f"PRAGMA index_info('{idx[1]}')")
            structure['index_info'][idx[1]] = cursor.fetchall()
        
        # Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list('{table}')")
        structure['foreign_keys'] = cursor.fetchall()
        
        return structure
    
    def compare_structures(self) -> Dict[str, Dict]:
        """Compare database structures and optimizations."""
        differences = {}
        
        # Get all tables
        self.orig_cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = [table[0] for table in self.orig_cursor.fetchall()]
        
        for table in tables:
            orig_structure = self.get_table_structure(self.orig_cursor, table)
            recon_structure = self.get_table_structure(self.recon_cursor, table)
            
            # Compare and store differences
            differences[table] = {
                'index_changes': self._compare_indexes(orig_structure, recon_structure),
                'constraint_changes': self._compare_constraints(orig_structure, recon_structure)
            }
        
        return differences
    
    def _compare_indexes(self, orig: Dict, recon: Dict) -> Dict:
        """Compare index structures between original and reconstructed."""
        changes = {
            'added': [],
            'removed': [],
            'modified': []
        }
        
        orig_idx_names = {idx[1] for idx in orig['indexes']}
        recon_idx_names = {idx[1] for idx in recon['indexes']}
        
        # Find added and removed indexes
        changes['added'] = list(recon_idx_names - orig_idx_names)
        changes['removed'] = list(orig_idx_names - recon_idx_names)
        
        # Compare common indexes
        common_indexes = orig_idx_names & recon_idx_names
        for idx_name in common_indexes:
            orig_info = orig['index_info'][idx_name]
            recon_info = recon['index_info'][idx_name]
            if orig_info != recon_info:
                changes['modified'].append(idx_name)
        
        return changes
    
    def _compare_constraints(self, orig: Dict, recon: Dict) -> Dict:
        """Compare constraints between original and reconstructed."""
        return {
            'foreign_keys': {
                'original': len(orig['foreign_keys']),
                'reconstructed': len(recon['foreign_keys']),
                'differences': len(orig['foreign_keys']) != len(recon['foreign_keys'])
            }
        }
    
    def analyze_query_plan(self, query: str) -> Tuple[List, List]:
        """Analyze and compare query execution plans."""
        # Get original query plan
        self.orig_cursor.execute(f"EXPLAIN QUERY PLAN {query}")
        orig_plan = self.orig_cursor.fetchall()
        
        # Get reconstructed query plan
        self.recon_cursor.execute(f"EXPLAIN QUERY PLAN {query}")
        recon_plan = self.recon_cursor.fetchall()
        
        return orig_plan, recon_plan
    
    def analyze_test_queries(self) -> Dict:
        """Analyze the test queries used in the compression test."""
        test_queries = [
            """
            SELECT c.CustomerName, COUNT(o.OrderID) as OrderCount
            FROM Customers c
            JOIN Orders o ON c.CustomerID = o.CustomerID
            GROUP BY c.CustomerID
            ORDER BY OrderCount DESC
            LIMIT 5
            """,
            """
            SELECT p.ProductName, 
                   COUNT(od.OrderID) as TimesSold,
                   SUM(od.Quantity) as TotalQuantity
            FROM Products p
            JOIN OrderDetails od ON p.ProductID = od.ProductID
            GROUP BY p.ProductID
            HAVING TimesSold > 5
            ORDER BY TotalQuantity DESC
            """,
            """
            SELECT c.CategoryName,
                   COUNT(DISTINCT p.ProductID) as ProductCount,
                   AVG(p.Price) as AvgPrice
            FROM Categories c
            JOIN Products p ON c.CategoryID = p.CategoryID
            GROUP BY c.CategoryID
            ORDER BY ProductCount DESC
            """
        ]
        
        analysis = {}
        for i, query in enumerate(test_queries, 1):
            orig_plan, recon_plan = self.analyze_query_plan(query)
            analysis[f'query_{i}'] = {
                'original_plan': orig_plan,
                'reconstructed_plan': recon_plan,
                'plan_differences': orig_plan != recon_plan
            }
        
        return analysis
    
    def close(self):
        """Close database connections."""
        self.orig_conn.close()
        self.recon_conn.close()

def main():
    """Analyze optimization differences."""
    # Setup paths
    base_dir = Path("test_results/compression_test_20240331")
    original_db = base_dir / "northwind.db"
    reconstructed_db = base_dir / "northwind_reconstructed.db"
    
    # Initialize analyzer
    analyzer = OptimizationAnalyzer(str(original_db), str(reconstructed_db))
    
    # Compare structures
    print("\nAnalyzing structural differences...")
    differences = analyzer.compare_structures()
    
    for table, diff in differences.items():
        print(f"\nTable: {table}")
        print("Index Changes:")
        print(f"  Added: {diff['index_changes']['added']}")
        print(f"  Removed: {diff['index_changes']['removed']}")
        print(f"  Modified: {diff['index_changes']['modified']}")
        print("Constraint Changes:")
        print(f"  Foreign Keys: {diff['constraint_changes']['foreign_keys']}")
    
    # Analyze query plans
    print("\nAnalyzing query execution plans...")
    query_analysis = analyzer.analyze_test_queries()
    
    for query_name, analysis in query_analysis.items():
        print(f"\n{query_name}:")
        print("Original Plan:")
        for step in analysis['original_plan']:
            print(f"  {step}")
        print("Reconstructed Plan:")
        for step in analysis['reconstructed_plan']:
            print(f"  {step}")
        print(f"Plans Differ: {analysis['plan_differences']}")
    
    analyzer.close()

if __name__ == "__main__":
    main() 