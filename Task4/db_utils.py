"""
Database utility functions for Task4: AI Conference Papers to Firms
"""

import os
import pandas as pd
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
from dotenv import load_dotenv
from contextlib import contextmanager

# Load environment variables
load_dotenv()


class MySQLDatabase:
    """MySQL Database connection handler"""

    def __init__(self):
        """Initialize database configuration from environment variables"""
        self.config = {
            'host': os.getenv('DB_HOST', 'misr.sauder.ubc.ca'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'database': os.getenv('DB_NAME', 'ai_science'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

        if not self.config['user'] or not self.config['password']:
            raise ValueError("Database credentials not found. Please set DB_USER and DB_PASSWORD in .env file")

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections

        Usage:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        connection = None
        try:
            connection = mysql.connector.connect(**self.config)
            if connection.is_connected():
                yield connection
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()

    def get_sqlalchemy_engine(self):
        """
        Get SQLAlchemy engine for pandas integration

        Returns:
            sqlalchemy.engine.Engine
        """
        connection_string = (
            f"mysql+mysqlconnector://{self.config['user']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )
        return create_engine(connection_string)

    def execute_query(self, query, params=None):
        """
        Execute a SELECT query and return results as a pandas DataFrame

        Args:
            query (str): SQL query
            params (tuple, optional): Query parameters

        Returns:
            pd.DataFrame: Query results
        """
        with self.get_connection() as conn:
            return pd.read_sql(query, conn, params=params)

    def execute_update(self, query, params=None):
        """
        Execute an INSERT, UPDATE, or DELETE query

        Args:
            query (str): SQL query
            params (tuple, optional): Query parameters

        Returns:
            int: Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount

    def get_tables(self):
        """
        Get list of all tables in the database

        Returns:
            list: Table names
        """
        query = "SHOW TABLES"
        df = self.execute_query(query)
        return df.iloc[:, 0].tolist()

    def get_table_info(self, table_name):
        """
        Get column information for a specific table

        Args:
            table_name (str): Name of the table

        Returns:
            pd.DataFrame: Table structure information
        """
        query = f"DESCRIBE {table_name}"
        return self.execute_query(query)

    def bulk_insert_dataframe(self, df, table_name, if_exists='append', chunksize=1000):
        """
        Insert a pandas DataFrame into a MySQL table

        Args:
            df (pd.DataFrame): DataFrame to insert
            table_name (str): Target table name
            if_exists (str): How to behave if table exists ('fail', 'replace', 'append')
            chunksize (int): Number of rows to insert at a time

        Returns:
            int: Number of rows inserted
        """
        engine = self.get_sqlalchemy_engine()
        rows_inserted = df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize
        )
        return rows_inserted if rows_inserted else len(df)


def test_connection():
    """Test database connection"""
    try:
        db = MySQLDatabase()
        print("Testing database connection...")

        with db.get_connection() as conn:
            if conn.is_connected():
                db_info = conn.get_server_info()
                print(f"✓ Successfully connected to MySQL Server version {db_info}")

                cursor = conn.cursor()
                cursor.execute("SELECT DATABASE();")
                database = cursor.fetchone()[0]
                print(f"✓ Connected to database: {database}")

                # List tables
                tables = db.get_tables()
                print(f"✓ Found {len(tables)} tables in database")
                if tables:
                    print(f"  Tables: {', '.join(tables[:5])}")
                    if len(tables) > 5:
                        print(f"  ... and {len(tables) - 5} more")

                return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
