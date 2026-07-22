import logging
from typing import Dict, Any, Union
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.tools import tool
from Database.connection import get_db_engine
from Middlewares.HIL_Middleware import HILManager, PendingOperation

logger = logging.getLogger(__name__)

@tool
def execute_ddl_operation(query: str, impact_description: str) -> str:
    """
    Executes a Schema Modification (DDL) query against the database 
    (e.g., CREATE TABLE, ALTER TABLE, ADD INDEX). 
    WARNING: This tool triggers a mandatory Human-in-the-Loop security interrupt 
    before executing.
    
    Args:
        query: The raw SQL DDL statement to execute.
        impact_description: A clear explanation of why this change is needed and what it affects.
    """
    try:
        clean_query = query.strip().rstrip(";")
        upper_query = clean_query.upper()
        
        # 1. Strict Security Block: Never allow dropping entire databases or truncation
        catastrophic_keywords = ("DROP DATABASE", "TRUNCATE", "DROP SCHEMA")
        if any(upper_query.startswith(kw) for kw in catastrophic_keywords):
            return "SECURITY_ERROR: Catastrophic actions like dropping databases or truncating tables are permanently blocked."

        # 2. Identify target table or action type
        op_type = "ALTER"
        if upper_query.startswith("CREATE"):
            op_type = "CREATE"
        elif upper_query.startswith("DROP"):
            op_type = "DROP"

        # 3. Format the HIL approval request
        pending_op = PendingOperation(
            operation_type=op_type,
            raw_query=clean_query,
            impact_analysis=impact_description,
            target_table="Database Schema Structure"
        )
        
        approval_prompt = HILManager.format_approval_request(pending_op)
        print(approval_prompt)

        # 4. Await Admin Decision (In CLI mode; in web mode this pauses via LangGraph checkpointer)
        # Note: When integrated into the full LangGraph state machine, this input() 
        # is replaced by the graph's interrupt mechanism.
        admin_input = input("Enter approval decision: ")
        is_approved = HILManager.process_decision(admin_input)

        if not is_approved:
            return "OPERATION_ABORTED: Schema modification was rejected or canceled by the administrator."

        # 5. Execute the DDL query if approved
        engine = get_db_engine()
        with engine.connect() as connection:
            # DDL operations usually require a commit or auto-commit behavior
            connection.execute(text(clean_query))
            connection.commit()
            
            logger.info(f"Successfully executed DDL operation: {clean_query}")
            return f"SUCCESS: Schema modification successfully executed and committed."

    except SQLAlchemyError as e:
        logger.error(f"DDL Execution Error: {str(e)}")
        return f"DDL_ERROR: {str(e)}\nPlease check your syntax."