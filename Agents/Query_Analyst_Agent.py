import os
import logging
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()

from State.Schema import DBAState

from Tools.Query_Analysis_Tools import get_query_execution_plan, get_table_indexes

logger = logging.getLogger(__name__)

QUERY_ANALYST_TOOLS = [
    get_query_execution_plan,
    get_table_indexes,
]

QUERY_ANALYST_SYSTEM_PROMPT = """
You are the Query Analyst agent, part of a DBA multi-agent system.

Your job is to diagnose and improve SQL query performance. You do NOT fetch or modify data.

1. Use get_query_execution_plan to see how the database will execute a given query
   (cost, rows examined, full table scans, index usage).
2. Use get_table_indexes to check what indexes already exist on the relevant tables,
   so you don't suggest redundant indexes and can spot missing ones.
3. Based on the execution plan and existing indexes, give the user a clear review:
   - Point out full table scans, missing indexes, or inefficient joins/filters.
   - Suggest specific improvements (e.g. "add an index on orders.customer_id").
   - If the query already looks efficient, say so plainly.

Only call one tool at a time. Once you have enough information to give a complete
review, respond with a final plain-text answer and do NOT call any more tools — this
hands control back to the supervisor and updates the state status to SUCCESS.
"""


def query_analyst_agent_node(state: MessagesState) -> dict:
    """
    Low-tier reasoning LLM bound to the query analyst's toolset.
    Decides whether to call a tool or produce a final review.
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(QUERY_ANALYST_TOOLS)

    messages = [SystemMessage(content=QUERY_ANALYST_SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)

    logger.info(f"Query Analyst agent response: {response.content[:200] if response.content else '[tool call]'}")
    
    # Check if the agent is outputting a final answer (no tool calls)
    # If it has a response and no tool calls, report SUCCESS back to the master state graph.
    is_final_answer = not bool(getattr(response, "tool_calls", None))
    agent_status = "SUCCESS" if is_final_answer else "PENDING"

    return {
        "messages": [response],
        "agent_status": agent_status
    }


def build_query_analyst_graph():
    """
    Builds the Query Analyst sub-graph:
      agent -> (tool_calls?) -> tools -> agent -> ... -> END
      agent -> (no tool_calls) -> END

    The compiled graph is callable as a single node, so it drops directly
    into the DBA supervisor graph as `query_analyzer_node`.
    """
    workflow = StateGraph(DBAState)

    workflow.add_node("agent", query_analyst_agent_node)
    workflow.add_node("tools", ToolNode(QUERY_ANALYST_TOOLS))

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    workflow.add_edge("tools", "agent")

    return workflow.compile()


# Compiled graph — this is what gets imported into the main DBA graph as query_analyzer_node.
query_analyst_graph = build_query_analyst_graph()