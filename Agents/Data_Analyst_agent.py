import os
import logging
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()

from State.Schema import DBAState

from Database.Database_Tools import list_database_tables, get_table_schemas, execute_read_query
from Tools.Data_Analysis_Tools import generate_chart, generate_report

logger = logging.getLogger(__name__)

# Tools available to the SQL Analyst agent.
# NOTE: execute_sql_query is intentionally excluded here — it allows writes/DDL
# and must stay reserved for the schema_admin agent (human-approval gated).
SQL_ANALYST_TOOLS = [
    list_database_tables,
    get_table_schemas,
    execute_read_query,
    generate_chart,
    generate_report,
]

SQL_ANALYST_SYSTEM_PROMPT = """
You are the SQL Analyst agent, part of a DBA multi-agent system.
The database is MySQL (not SQLite, not PostgreSQL). Always use MySQL-compatible syntax:
- Extract year: YEAR(column)
- Extract month: MONTH(column)
- Date literals: 'YYYY-MM-DD'
- Do NOT use strftime, date_trunc, or other non-MySQL functions.

Your job:
1. Explore the database using list_database_tables and get_table_schemas to understand structure BEFORE writing queries.
2. Use execute_read_query to fetch data with SELECT statements only. You cannot modify data.
3. Use generate_chart to visualize results when the user wants to see trends or comparisons.
4. Use generate_report to produce a written summary of your findings when asked for analysis/insights.

Only call one tool at a time. Once you have enough information to fully answer the user's
request, respond with a final answer and do NOT call any more tools.

FORMAT FOR YOUR FINAL ANSWER (not for tool calls):
- Use Markdown.
- If listing tables/schemas, use a bulleted list or a Markdown table — one row/bullet per table.
- If listing columns, format as: `table_name: col1, col2, col3 (FK: col -> other_table)`.
- Keep prose minimal; prefer structured lists over paragraphs.
- Do not repeat the full schema if the user only asked a narrow question.
"""


def sql_agent_node(state: MessagesState) -> dict:
    """
    Low-tier reasoning LLM bound to the SQL analyst's toolset.
    Decides whether to call a tool or produce a final answer.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # lighter/cheaper model for tool-routing within the sub-agent
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(SQL_ANALYST_TOOLS)

    messages = [SystemMessage(content=SQL_ANALYST_SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)

    logger.info(f"SQL Analyst agent response: {response.content[:200] if response.content else '[tool call]'}")
    
    # Check if the agent is outputting a final answer (no tool calls)
    # If it has a response and no tool calls, report SUCCESS back to the master state graph.
    is_final_answer = not bool(getattr(response, "tool_calls", None))
    agent_status = "SUCCESS" if is_final_answer else "PENDING"

    return {
        "messages": [response],
        "agent_status": agent_status
    }


def build_sql_analyst_graph():
    """
    Builds the SQL Analyst sub-graph:
      agent -> (tool_calls?) -> tools -> agent -> ... -> END
      agent -> (no tool_calls) -> END

    The compiled graph is itself callable as a single node, so it can be
    dropped directly into the DBA supervisor graph as `sql_analyst_node`.
    """
    workflow = StateGraph(DBAState)

    workflow.add_node("agent", sql_agent_node)
    workflow.add_node("tools", ToolNode(SQL_ANALYST_TOOLS))

    workflow.add_edge(START, "agent")

    # tools_condition inspects the last message for tool_calls:
    # routes to "tools" if present, otherwise to END
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # After executing a tool, always return to the agent to decide the next step
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# Compiled graph — this is what gets imported into the main DBA graph as sql_analyst_node.
sql_analyst_graph = build_sql_analyst_graph()