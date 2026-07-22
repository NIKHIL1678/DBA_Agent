import json
import logging
from typing import List, Dict, Any
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.tools import tool
from Database.connection import get_db_engine

logger = logging.getLogger(__name__)

@tool
def get_query_execution_plan(query: str) -> str:
    """
    Analyzes a SQL query by running EXPLAIN FORMAT=JSON on it.
    This does NOT return the query data. It returns MySQL's execution plan,
    showing query cost, rows examined, and whether indexes were used or ignored.
    Use this to diagnose slow queries or check for full table scans.
    
    Args:
        query: The raw SQL query (e.g., SELECT, UPDATE, DELETE) to analyze.
    """
    try:
        # Strip trailing semicolons to cleanly prepend EXPLAIN
        clean_query = query.strip().rstrip(";")
        
        # Security: Prevent chaining multiple queries or explaining a DROP/ALTER
        forbidden_keywords = ("ALTER", "DROP", "CREATE", "TRUNCATE", "REPLACE")
        if clean_query.upper().startswith(forbidden_keywords):
            return "ERROR: EXPLAIN is only supported for SELECT, INSERT, UPDATE, and DELETE operations."

        explain_query = f"EXPLAIN FORMAT=JSON {clean_query}"
        engine = get_db_engine()
        
        with engine.connect() as connection:
            result = connection.execute(text(explain_query))
            # MySQL EXPLAIN FORMAT=JSON returns a single row with a single JSON column
            plan_json = result.scalar()
            
            return f"Execution Plan:\n{plan_json}"
            
    except SQLAlchemyError as e:
        logger.error(f"Explain Plan Error: {str(e)}")
        return f"ERROR analyzing query: {str(e)}\nCheck syntax and ensure tables exist."

@tool
def get_table_indexes(table_name: str) -> str:
    """
    Retrieves a list of all current indexes on a specified table.
    Use this tool when you discover a slow query to see if the necessary
    indexes already exist, or to avoid suggesting duplicate indexes.
    
    Args:
        table_name: The name of the database table to inspect.
    """
    try:
        engine = get_db_engine()
        inspector = inspect(engine)
        
        if not inspector.has_table(table_name):
            return f"ERROR: Table '{table_name}' does not exist."
            
        indexes = inspector.get_indexes(table_name)
        
        if not indexes:
            return f"Table '{table_name}' has NO indexes (other than primary key if defined)."
            
        index_details = []
        for idx in indexes:
            name = idx.get('name')
            columns = idx.get('column_names', [])
            unique = idx.get('unique', False)
            
            idx_str = f"- Index: '{name}' | Columns: {columns} | Unique: {unique}"
            index_details.append(idx_str)
            
        return f"Indexes for table '{table_name}':\n" + "\n".join(index_details)
        
    except SQLAlchemyError as e:
        logger.error(f"Index Retrieval Error: {str(e)}")
        return f"ERROR retrieving indexes: {str(e)}"