import logging
from dataclasses import dataclass
from typing import Optional, Literal

logger = logging.getLogger(__name__)

@dataclass
@dataclass
class PendingOperation:
    """Strict data structure representing a high-risk database operation."""
    operation_type: Literal["UPDATE", "DELETE", "INSERT", "DROP", "ALTER", "CREATE"]
    raw_query: str
    impact_analysis: str
    target_table: str

class HILManager:
    """
    Manages formatting and handling for Human-in-the-Loop interventions 
    during high-risk database executions.
    """
    
    @staticmethod
    def format_approval_request(operation: PendingOperation) -> str:
        """
        Generates a standardized alert payload. In a web environment, 
        this would be serialized to JSON and sent to a frontend admin panel.
        """
        alert = (
            f"\n{'='*60}\n"
            f"🚨 HIGH PRIVILEGE OPERATION REQUIRES APPROVAL 🚨\n"
            f"{'='*60}\n"
            f"Action:       {operation.operation_type}\n"
            f"Target Table: {operation.target_table}\n"
            f"Query:        {operation.raw_query}\n"
            f"Impact:       {operation.impact_analysis}\n"
            f"{'='*60}\n"
        )
        logger.warning(f"HIL interrupt triggered for {operation.operation_type} on {operation.target_table}")
        return alert

    @staticmethod
    def process_decision(decision_input: str) -> bool:
        """
        Parses human input defensively. 
        Returns True for execution, False for rollback/rejection.
        """
        if not decision_input:
            logger.info("HIL: Empty decision received. Defaulting to REJECT.")
            return False
            
        clean_input = str(decision_input).strip().upper()
        
        if clean_input in ["APPROVE", "YES", "Y", "EXECUTE"]:
            logger.info("HIL: Operation APPROVED by administrator.")
            return True
            
        logger.info(f"HIL: Operation REJECTED. User input: {clean_input}")
        return False