"""
Interactive database explorer for comparing original and reconstructed databases.
"""

import sqlite3
from tabulate import tabulate
import sys
from pathlib import Path

class DatabaseExplorer:
    def __init__(self, original_db: str, reconstructed_db: str):
        """Initialize database connections."""
        self.orig_conn = sqlite3.connect(original_db)
        self.recon_conn = sqlite3.connect(reconstructed_db)
        self.orig_cursor = self.orig_conn.cursor()
        self.recon_cursor = self.recon_conn.cursor()
    
    def get_tables(self) -> list:
        """Get list of tables in the database."""
        self.orig_cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        return [table[0] for table in self.orig_cursor.fetchall()]
    
    def compare_table(self, table_name: str) -> tuple:
        """Compare data in a table between original and reconstructed databases."""
        # Get original data
        self.orig_cursor.execute(f"SELECT * FROM {table_name}")
        orig_data = self.orig_cursor.fetchall()
        
        # Get column names
        self.orig_cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in self.orig_cursor.fetchall()]
        
        # Get reconstructed data
        self.recon_cursor.execute(f"SELECT * FROM {table_name}")
        recon_data = self.recon_cursor.fetchall()
        
        return columns, orig_data, recon_data
    
    def get_table_stats(self, table_name: str) -> dict:
        """Get statistics about a table."""
        stats = {}
        
        # Row count
        self.orig_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        stats['row_count'] = self.orig_cursor.fetchone()[0]
        
        # Column info
        self.orig_cursor.execute(f"PRAGMA table_info({table_name})")
        columns = self.orig_cursor.fetchall()
        stats['column_count'] = len(columns)
        stats['columns'] = [col[1] for col in columns]
        
        # Index info
        self.orig_cursor.execute(f"PRAGMA index_list({table_name})")
        indexes = self.orig_cursor.fetchall()
        stats['index_count'] = len(indexes)
        stats['indexes'] = [idx[1] for idx in indexes]
        
        # Foreign key info
        self.orig_cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = self.orig_cursor.fetchall()
        stats['foreign_key_count'] = len(foreign_keys)
        stats['foreign_keys'] = [(fk[3], fk[2], fk[4]) for fk in foreign_keys]  # (from, to_table, to_col)
        
        return stats
    
    def close(self):
        """Close database connections."""
        self.orig_conn.close()
        self.recon_conn.close()

def main():
    """Interactive database explorer."""
    # Setup paths
    base_dir = Path("test_results/compression_test_20240331")
    original_db = base_dir / "northwind.db"
    reconstructed_db = base_dir / "northwind_reconstructed.db"
    
    # Initialize explorer
    explorer = DatabaseExplorer(str(original_db), str(reconstructed_db))
    
    while True:
        print("\nDatabase Explorer")
        print("================")
        print("1. List tables")
        print("2. Compare table data")
        print("3. Show table statistics")
        print("4. Run custom query")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            tables = explorer.get_tables()
            print("\nAvailable Tables:")
            for i, table in enumerate(tables, 1):
                print(f"{i}. {table}")
        
        elif choice == "2":
            tables = explorer.get_tables()
            print("\nAvailable Tables:")
            for i, table in enumerate(tables, 1):
                print(f"{i}. {table}")
            
            table_idx = int(input("\nEnter table number: ")) - 1
            if 0 <= table_idx < len(tables):
                table_name = tables[table_idx]
                columns, orig_data, recon_data = explorer.compare_table(table_name)
                
                print(f"\nOriginal {table_name}:")
                print(tabulate(orig_data, headers=columns, tablefmt="grid"))
                
                print(f"\nReconstructed {table_name}:")
                print(tabulate(recon_data, headers=columns, tablefmt="grid"))
                
                if orig_data == recon_data:
                    print("\n✅ Data matches exactly!")
                else:
                    print("\n❌ Data differs!")
        
        elif choice == "3":
            tables = explorer.get_tables()
            print("\nAvailable Tables:")
            for i, table in enumerate(tables, 1):
                print(f"{i}. {table}")
            
            table_idx = int(input("\nEnter table number: ")) - 1
            if 0 <= table_idx < len(tables):
                table_name = tables[table_idx]
                stats = explorer.get_table_stats(table_name)
                
                print(f"\nStatistics for {table_name}:")
                print(f"Rows: {stats['row_count']}")
                print(f"Columns ({stats['column_count']}):")
                for col in stats['columns']:
                    print(f"  - {col}")
                
                print(f"\nIndexes ({stats['index_count']}):")
                for idx in stats['indexes']:
                    print(f"  - {idx}")
                
                print(f"\nForeign Keys ({stats['foreign_key_count']}):")
                for fk in stats['foreign_keys']:
                    print(f"  - {fk[0]} → {fk[1]}.{fk[2]}")
        
        elif choice == "4":
            query = input("\nEnter SQL query: ")
            try:
                print("\nOriginal Database:")
                explorer.orig_cursor.execute(query)
                orig_results = explorer.orig_cursor.fetchall()
                if explorer.orig_cursor.description:
                    headers = [col[0] for col in explorer.orig_cursor.description]
                    print(tabulate(orig_results, headers=headers, tablefmt="grid"))
                
                print("\nReconstructed Database:")
                explorer.recon_cursor.execute(query)
                recon_results = explorer.recon_cursor.fetchall()
                if explorer.recon_cursor.description:
                    headers = [col[0] for col in explorer.recon_cursor.description]
                    print(tabulate(recon_results, headers=headers, tablefmt="grid"))
                
                if orig_results == recon_results:
                    print("\n✅ Results match exactly!")
                else:
                    print("\n❌ Results differ!")
            
            except sqlite3.Error as e:
                print(f"\nError executing query: {e}")
        
        elif choice == "5":
            explorer.close()
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main() 