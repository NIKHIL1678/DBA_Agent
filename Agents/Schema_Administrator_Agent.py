import os
import logging
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()

from State.Schema import DBAState

from Tools.Schema_Administrator_Tools import execute_ddl_operation

logger = logging.getLogger(__name__)

SCHEMA_ADMIN_TOOLS = [
    execute_ddl_operation,
]

SCHEMA_ADMIN_SYSTEM_PROMPT = """
You are the Schema Administrator agent, part of a DBA multi-agent system.

Your job is to handle high-privilege schema modifications (CREATE, ALTER, DROP tables,
adding indexes, etc.) via the execute_ddl_operation tool.

Rules:
1. Never invent a DDL statement without being reasonably confident about the table
   structure it affects. If you're unsure, ask the user for clarification instead of
   guessing.
2. Always provide a clear, honest impact_description when calling execute_ddl_operation —
   explain what the change does and what it affects. This is shown to a human administrator
   who must approve it before anything runs.
3. Every call to execute_ddl_operation triggers a mandatory human approval step. If the
   operation is rejected, tell the user plainly and do not retry the same statement
   without a new instruction from them.

Only call one tool at a time. Once the operation is complete (approved+executed, or
rejected), respond with a final plain-text summary and do NOT call any more tools —
this hands control back to the supervisor.
"""


def schema_admin_agent_node(state: MessagesState) -> dict:
    """
    Low-tier reasoning LLM bound to the schema admin's toolset.
    Decides whether to call the DDL tool or produce a final summary.
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(SCHEMA_ADMIN_TOOLS)

    messages = [SystemMessage(content=SCHEMA_ADMIN_SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)

    logger.info(f"Schema Admin agent response: {response.content[:200] if response.content else '[tool call]'}")
    return {"messages": [response]}


def build_schema_admin_graph():
    """
    Builds the Schema Administrator sub-graph:
      agent -> (tool_calls?) -> tools -> agent -> ... -> END
      agent -> (no tool_calls) -> END

    The compiled graph is callable as a single node, so it drops directly
    into the DBA supervisor graph as `schema_admin_node`.
    """
    workflow = StateGraph(DBAState)

    workflow.add_node("agent", schema_admin_agent_node)
    workflow.add_node("tools", ToolNode(SCHEMA_ADMIN_TOOLS))

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


# Compiled graph — this is what gets imported into the main DBA graph as schema_admin_node.
schema_admin_graph = build_schema_admin_graph()