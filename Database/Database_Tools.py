import json
import logging
from typing import List
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.tools import tool
from Database.connection import get_db_engine

logger = logging.getLogger(__name__)

@tool
def list_database_tables() -> str:
    """
    Returns a comma-separated string of all table names in the database. 
    Use this first to figure out which tables might be relevant to the user's query.
    """
    try:
        engine = get_db_engine()
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if not tables:
            return "No tables found in the database."
            
        return f"Found {len(tables)} tables: {', '.join(tables)}"
    except Exception as e:
        return f"Error fetching tables: {str(e)}"

@tool
def get_table_schemas(table_names: List[str]) -> str:
    """
    Takes a list of specific table names and returns their columns, data types, 
    and foreign keys. Use this to understand the structure of specific tables 
    BEFORE writing a SQL query.
    """
    try:
        engine = get_db_engine()
        inspector = inspect(engine)
        schema_lines = []

        for table in table_names:
            if not inspector.has_table(table):
                schema_lines.append(f"Table '{table}' does not exist.")
                continue

            columns = inspector.get_columns(table)
            col_details = [f"{col['name']} ({col['type']})" for col in columns]
            
            fks = inspector.get_foreign_keys(table)
            fk_details = [
                f"{fk['constrained_columns'][0]} -> {fk['referred_table']}({fk['referred_columns'][0]})"
                for fk in fks
            ]

            schema_lines.append(f"Table: {table}")
            schema_lines.append(f"  Columns: {', '.join(col_details)}")
            if fk_details:
                schema_lines.append(f"  Foreign Keys: {', '.join(fk_details)}")
            schema_lines.append("")

        result = "\n".join(schema_lines).strip()
        # Prevent returning an empty string if something weird happens
        return result if result else "No schema information found."
    except Exception as e:
        return f"Error fetching schemas: {str(e)}"

@tool
def execute_read_query(query: str) -> str:
    """
    Executes a read-only (SELECT) SQL query against the database to fetch data.
    This tool is strictly for finding and analyzing data. It will block any modifying operations.
    """
    try:
        # 1. Security Check: Block modifying queries
        forbidden_keywords = ("INSERT", "UPDATE", "DELETE", "ALTER", "DROP", "CREATE", "TRUNCATE", "REPLACE")
        if query.strip().upper().startswith(forbidden_keywords):
            return "SECURITY_ERROR: This agent is only allowed to execute SELECT queries. Modifying operations are blocked."

        # 2. Set up connection using our connection.py manager
        engine = get_db_engine()
        
        # 3. Execute the query and fetch the data
        with engine.connect() as connection:
            result = connection.execute(text(query))
            data = [dict(row._mapping) for row in result.fetchall()]
            
            # 4. Handle empty results safely
            if not data:
                return "Query executed successfully, but returned 0 rows."
            
            # Safety limit for output size to prevent token explosion on massive tables
            if len(data) > 100:
                logger.warning(f"Query returned {len(data)} rows. Truncating to 100 for token safety.")
                data = data[:100]
                data.append({"WARNING": f"Results truncated for LLM context limits. Total rows were greater than 100."})
            
            # 5. Convert to JSON string (default=str handles Decimals/Dates that JSON normally chokes on)
            return json.dumps(data, default=str)
            
    except SQLAlchemyError as e:
        logger.error(f"SQL Read Execution Error: {str(e)}")
        return f"SQL_ERROR: {str(e)}\nPlease review the schema and correct your query."

@tool
def execute_sql_query(query: str) -> str:
    """
    Executes a raw SQL query against the database.
    WARNING: Only use this after verifying the query syntax and schema.
    """
    try:
        engine = get_db_engine()
        with engine.connect() as connection:
            result = connection.execute(text(query))
            
            # Check for modifying queries to apply commits
            is_write = query.strip().upper().startswith(
                ("INSERT", "UPDATE", "DELETE", "ALTER", "DROP", "CREATE")
            )
            
            if is_write:
                connection.commit()
                return f"SUCCESS: Query executed. Rows affected: {result.rowcount}"
            
            # Read operations
            data = [dict(row._mapping) for row in result.fetchall()]
            
            if not data:
                return "Query executed successfully, but returned 0 rows."
            
            # Safety limit for output size to prevent token explosion on massive tables
            if len(data) > 100:
                data = data[:100]
                data.append({"WARNING": f"Results truncated. Total rows were greater than 100."})
                
            return json.dumps(data, default=str)
            
    except SQLAlchemyError as e:
        logger.error(f"SQL Execution Error: {str(e)}")
        return f"SQL_ERROR: {str(e)}\nPlease review the schema and correct your query."