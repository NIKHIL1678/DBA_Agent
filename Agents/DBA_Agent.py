import os
import logging
from typing import Literal, List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from Middlewares.PII_filters import PIIFilter

from rich.console import Console
from rich.markdown import Markdown

console = Console(width=100)

from State.Schema import DBAState

# Import the COMPILED sub-agent graphs (Runnables)
from Agents.Data_Analyst_agent import sql_analyst_graph as sql_analyst_node
from Agents.Query_Analyst_Agent import query_analyst_graph as query_analyzer_node
from Agents.Schema_Administrator_Agent import schema_admin_graph as schema_admin_node

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """
You are the Master Supervisor of an elite Database Administrator (DBA) multi-agent system.
Your job is to analyze the user's request and route it to the correct specialized sub-agent, or respond directly if it's a general question.

Available Sub-Agents:
1. 'sql_analyst': Use this agent for writing SELECT queries, exploring database tables/schemas, fetching data, generating charts, and writing data analysis reports.
2. 'query_analyzer': Use this agent for checking slow queries, running EXPLAIN plans, analyzing index usage, and optimizing query performance.
3. 'schema_admin': Use this agent for high-privilege DDL modifications (CREATE, ALTER, DROP tables or changing structure). WARNING: This triggers human approvals.

CRITICAL INSTRUCTIONS FOR ROUTING:
- If a sub-agent has already answered or `agent_status` is "SUCCESS", you MUST output "FINISH".
- Do NOT route back to a sub-agent if the task for the current user request is already complete.
- Only choose a sub-agent if there is an active, unfulfilled request from the user.

You must respond with ONLY ONE of the following routing choices:
- "sql_analyst"
- "query_analyzer"
- "schema_admin"
- "FINISH"
"""

MAX_SUPERVISOR_LOOPS = 4  # safety cap

def supervisor_node(state: DBAState) -> Dict[str, Any]:
    """
    Evaluates the current state and conversation history to determine
    which sub-agent should handle the next step.
    """
    loop_count = state.get("loop_count", 0) + 1
    agent_status = state.get("agent_status", "PENDING")

    # STRICT CHECK: If sub-agent completed successfully or loops hit max, force FINISH immediately.
    if agent_status == "SUCCESS" or agent_status == "ERROR" or loop_count > MAX_SUPERVISOR_LOOPS:
        logger.info(f"Supervisor short-circuiting to FINISH. Status: {agent_status}, Loops: {loop_count}")
        return {
            "next_agent": "FINISH",
            "loop_count": loop_count,
            "agent_status": agent_status
        }

    cleaned_messages = PIIFilter.apply_to_messages(state["messages"])

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        groq_api_key=groq_api_key
    )

    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + cleaned_messages
    response = llm.invoke(messages)

    raw_content = response.content
    if isinstance(raw_content, list):
        text_parts = [
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in raw_content
        ]
        content_str = "".join(text_parts)
    else:
        content_str = raw_content

    route = content_str.strip().lower()

    if "sql_analyst" in route:
        next_step = "sql_analyst"
    elif "query_analyzer" in route:
        next_step = "query_analyzer"
    elif "schema_admin" in route:
        next_step = "schema_admin"
    else:
        next_step = "FINISH"

    logger.info(f"Supervisor routing decision: {next_step} | Status: {agent_status}")

    if next_step == "FINISH":
        reply_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            groq_api_key=groq_api_key,
        )
        reply_system = (
            "You are the Master Supervisor of a DBA multi-agent system. "
            "Reply naturally, conversationally, and directly to the user's query or present the completed sub-agent findings cleanly. "
            "Do not use generic placeholders or execution templates."
        )
        reply_response = reply_llm.invoke([SystemMessage(content=reply_system)] + cleaned_messages)

        reply_content = reply_response.content
        if isinstance(reply_content, list):
            reply_content = "".join(
                block if isinstance(block, str) else str(block.get("text", ""))
                for block in reply_content
            )

        return {
            "next_agent": next_step, 
            "loop_count": loop_count, 
            "agent_status": "SUCCESS",
            "messages": [AIMessage(content=reply_content)]
        }

    return {"next_agent": next_step, "loop_count": loop_count, "agent_status": "PENDING"}

def should_continue(state: DBAState) -> Literal["sql_analyst", "query_analyzer", "schema_admin", "FINISH"]:
    """Conditional edge router based on supervisor output."""
    if state.get("agent_status") == "SUCCESS":
        return "FINISH"
    next_step = state.get("next_agent", "FINISH")
    if next_step in ("sql_analyst", "query_analyzer", "schema_admin", "FINISH"):
        return next_step
    return "FINISH"

def build_dba_graph():
    
    workflow = StateGraph(DBAState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("sql_analyst", sql_analyst_node)
    workflow.add_node("query_analyzer", query_analyzer_node)
    workflow.add_node("schema_admin", schema_admin_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "sql_analyst": "sql_analyst",
            "query_analyzer": "query_analyzer",
            "schema_admin": "schema_admin",
            "FINISH": END
        }
    )

    workflow.add_edge("sql_analyst", "supervisor")
    workflow.add_edge("query_analyzer", "supervisor")
    workflow.add_edge("schema_admin", "supervisor")

    return workflow.compile()


def run_cli():
    """
    Simple interactive CLI loop for testing the DBA multi-agent graph.
    """
    graph = build_dba_graph()
    conversation_state: Dict[str, Any] = {
        "messages": [], 
        "next_agent": "", 
        "loop_count": 0, 
        "agent_status": "PENDING"
    }

    print("=" * 60)
    print("DBA Multi-Agent System — Interactive CLI")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Exiting session. Goodbye!")
            break

        conversation_state["messages"].append(HumanMessage(content=user_input))
        conversation_state["loop_count"] = 0  
        conversation_state["agent_status"] = "PENDING"  # Reset status for new user turn

        try:
            result = graph.invoke(conversation_state)
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            print(f"\n[ERROR] Something went wrong: {e}")
            continue

        conversation_state = result

        messages = result.get("messages", [])
        last_ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        if last_ai_messages:
            latest = last_ai_messages[-1]
            content = latest.content
            if isinstance(content, list):
                content = "".join(
                    block if isinstance(block, str) else str(block.get("text", ""))
                    for block in content
                )

            console.print("-" * 60)
            console.print(Markdown(content.strip()))
            console.print("-" * 60)
        else:
            console.print("\n[Agent Response]: [No response generated]")


if __name__ == "__main__":
    run_cli()