# mcp_server.py

import sys
import io
from typing import cast

cast(io.TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8")
cast(io.TextIOWrapper, sys.stderr).reconfigure(encoding="utf-8")
cast(io.TextIOWrapper, sys.stdin).reconfigure(encoding="utf-8")

# ... rest of your imports (FastMCP, your tools, etc.) below this

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from utils.Logging_Config import setup_logging

load_dotenv()
setup_logging()

mcp = FastMCP("dba-agent")

from Database.Database_Tools import list_database_tables, get_table_schemas, execute_read_query
from Tools.Query_Analysis_Tools import get_query_execution_plan, get_table_indexes
from Tools.Data_Analysis_Tools import generate_chart, generate_report
from typing import List, Dict, Any, Union

@mcp.tool()
def ping() -> str:
    """Simple no-dependency test tool."""
    return "pong"

@mcp.tool()
def list_tables() -> str:
    """Returns a comma-separated string of all table names in the database."""
    return list_database_tables.invoke({})

@mcp.tool()
def get_schemas(table_names: List[str]) -> str:
    """Returns columns, data types, and foreign keys for the given table names."""
    return get_table_schemas.invoke({"table_names": table_names})

@mcp.tool()
def run_select_query(query: str) -> Union[List[Dict[str, Any]], str]:
    """Executes a read-only SELECT query and returns the results."""
    return execute_read_query.invoke({"query": query})

@mcp.tool()
def explain_query(query: str) -> str:
    """Runs EXPLAIN FORMAT=JSON on a query to show its execution plan."""
    return get_query_execution_plan.invoke({"query": query})

@mcp.tool()
def get_indexes(table_name: str) -> str:
    """Lists all current indexes on a specified table."""
    return get_table_indexes.invoke({"table_name": table_name})

@mcp.tool()
def make_chart(data: List[Dict[str, Any]], chart_type: str, x_column: str, y_column: str, title: str) -> str:
    """Generates a chart (bar/line/pie/scatter) from data and saves it as HTML."""
    return generate_chart.invoke({"data": data, "chart_type": chart_type, "x_column": x_column, "y_column": y_column, "title": title})

@mcp.tool()
def make_report(title: str, sql_query_used: str, data_summary: str, insights: str) -> str:
    """Generates a markdown analysis report and saves it to disk."""
    return generate_report.invoke({"title": title, "sql_query_used": sql_query_used, "data_summary": data_summary, "insights": insights})

# --- DDL: two-phase, no blocking input() ---
_pending_ddl = {}

@mcp.tool()
def propose_ddl_operation(query: str, impact_description: str) -> dict:
    """Stage a schema-changing (DDL) query for human approval. Does not execute anything yet."""
    op_id = str(len(_pending_ddl) + 1)
    _pending_ddl[op_id] = {"query": query, "impact": impact_description}
    return {"operation_id": op_id, "query": query, "impact": impact_description,
            "next_step": "Call approve_ddl_operation with this operation_id and decision='approve'/'reject'"}

@mcp.tool()
def approve_ddl_operation(operation_id: str, decision: str) -> str:
    """Approve or reject a previously proposed DDL operation."""
    op = _pending_ddl.pop(operation_id, None)
    if not op:
        return f"ERROR: no pending operation with id {operation_id}"
    if decision.strip().lower() not in ("approve", "yes", "y"):
        return "OPERATION_ABORTED: rejected by administrator."
    from Database.connection import get_db_engine
    from sqlalchemy import text
    engine = get_db_engine()
    with engine.connect() as conn:
        conn.execute(text(op["query"]))
        conn.commit()
    return f"SUCCESS: executed {op['query']}"

# --- Option B: whole supervisor graph as one tool ---
@mcp.tool()
def ask_dba_agent(user_query: str) -> str:
    """Send a natural-language DBA request to the full multi-agent system (routing, PII filtering, etc.)."""
    from Agents.DBA_Agent import build_dba_graph
    from langchain_core.messages import HumanMessage, AIMessage
    graph = build_dba_graph()
    result = graph.invoke({"messages": [HumanMessage(content=user_query)],
                            "next_agent": "", "loop_count": 0, "agent_status": "PENDING"})
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else "[no response]"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")