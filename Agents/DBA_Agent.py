"""
Multi-agent DBA system, sourcing its tools from the MCP server (mcp_server.py)
over streamable HTTP, instead of importing them directly from Database/Tools.

Architecture (mirrors the original Agents/DBA_Agent.py):

    supervisor --routes to  0. DB_Schema_Designer --back to--> supervisor
                            1. sql_analyst        --back to--> supervisor
                            2. query_analyzer     --back to--> supervisor
                            3. schema_admin       --back to--> supervisor
                            4. FINISH -> END

Differences from the original:
  - Tools come from MCP (client.get_tools()) instead of local imports.
  - Each sub-agent is built with `create_agent` (handles its own internal
    ReAct tool-calling loop), instead of a hand-rolled StateGraph +
    ToolNode + tools_condition.
  - PII redaction is handled by PIIMiddleware instead of a manual regex filter.
  - DDL approval is handled by HumanInTheLoopMiddleware (a real LangGraph
    interrupt) instead of a blocking input() call inside the tool itself.
  - Conversation memory persists across turns via a checkpointer + thread_id,
    instead of manually threading a conversation_state dict through a CLI loop.

Prerequisites:
    - mcp_server.py must already be running:  python mcp_server.py
      (listening on http://127.0.0.1:8000/mcp by default)
    - pip install langchain-mcp-adapters langchain langgraph langchain-groq

Run:
    python mcp_multi_agent.py
"""

import asyncio
import logging
from typing import Literal, Dict, Any, List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"

# The literal marker that tells us schema_designer has finished designing
# (vs. still mid-Q&A). Must match exactly what's in the prompt's Phase 3 header.
SCHEMA_DESIGN_COMPLETE_MARKER = "## SCHEMA DESIGN PROPOSAL"


# ---------------------------------------------------------------------------
# Shared graph state — same shape as the original DBAState
# ---------------------------------------------------------------------------

class DBAState(MessagesState):
    next_agent: str
    loop_count: int
    agent_status: str


MAX_SUPERVISOR_LOOPS = 6  # bumped up slightly since schema design can take several Q&A turns

SUPERVISOR_SYSTEM_PROMPT = """
You are the Master Supervisor of an elite Database Administrator (DBA) multi-agent system.
Your job is to analyze the user's request and route it to the correct specialized sub-agent,
or respond directly if it's a general question.

Available Sub-Agents:
1. 'DB_Schema_Designer': for open-ended "design me a database/table for X" requests.
   Gathers requirements conversationally (one question at a time), checks the
   existing schema, then produces a formatted schema design proposal.
   Route here FIRST for any new schema/table design request — NOT schema_admin.
2. 'sql_analyst': writing SELECT queries, exploring tables/schemas, fetching data,
   generating charts, and writing data analysis reports.
3. 'query_analyzer': checking slow queries, running EXPLAIN plans, analyzing index usage.
4. 'schema_admin': high-privilege DDL modifications (CREATE, ALTER, DROP tables).
   Route here when the user wants to execute/build a schema change — either
   a fully-specified change of their own, or a design already produced by
   DB_Schema_Designer in this conversation.
   WARNING: This triggers a mandatory human approval step.

CRITICAL INSTRUCTIONS FOR ROUTING:
- If a sub-agent has already answered or `agent_status` is "SUCCESS", you MUST output "FINISH".
- Do NOT route back to a sub-agent if the current request is already complete.
- If DB_Schema_Designer is still gathering requirements (has asked a question and
  is awaiting the user's answer), route back to "DB_Schema_Designer", NOT "FINISH".
- Only route to schema_admin to execute a design once DB_Schema_Designer has
  produced a complete "## SCHEMA DESIGN PROPOSAL" block, or the user gives a
  fully-specified DDL request directly without needing design help.

Respond with ONLY ONE of: "DB_Schema_Designer", "sql_analyst", "query_analyzer", "schema_admin", "FINISH"
"""


# ---------------------------------------------------------------------------
# Tool loading + splitting (MCP tools are a flat list; assign by name)
# ---------------------------------------------------------------------------

async def load_mcp_tool_groups() -> Dict[str, List[BaseTool]]:
    client = MultiServerMCPClient({
        "dba_agent": {
            "transport": "streamable_http",
            "url": MCP_SERVER_URL,
        }
    })
    all_tools = await client.get_tools()
    by_name = {t.name: t for t in all_tools}

    def pick(*names: str) -> List[BaseTool]:
        missing = [n for n in names if n not in by_name]
        if missing:
            logger.warning(f"Expected MCP tools not found on server: {missing}")
        return [by_name[n] for n in names if n in by_name]

    return {
        "DB_Schema_Designer": pick(
            "list_tables", "get_schemas", "get_indexes",
        ),
        "sql_analyst": pick(
            "list_tables", "get_schemas", "run_select_query",
            "make_chart", "make_report",
        ),
        "query_analyzer": pick("explain_query", "get_indexes"),
        "schema_admin": pick(
            "list_tables", "get_schemas", "get_indexes",
            "run_select_query",
            "propose_ddl_operation", "approve_ddl_operation",
        ),
    }


# ---------------------------------------------------------------------------
# Sub-agent builders — each is a fully self-contained create_agent
# (handles its own tool-calling loop internally, so by the time control
# returns to the supervisor, agent_status can simply be SUCCESS)...
# ...EXCEPT DB_Schema_Designer, whose node needs special PENDING/SUCCESS
# handling since it spans multiple conversational turns. See make_sub_agent_node.
# ---------------------------------------------------------------------------

DB_SCHEMA_DESIGNER_SYSTEM_PROMPT = """
You are the Database Schema Designer — an expert in relational database design,
normalization, indexing strategy, and translating business requirements into
production-ready schemas.

═══════════════════════════════════════════════════════════════
PHASE 0: CHECK EXISTING SCHEMA (always do this first, silently)
═══════════════════════════════════════════════════════════════
Before asking the user anything, call list_tables to see what already exists
in the database. If relevant tables are already present, call get_schemas
and get_indexes on them too. This tells you:
- Whether the user is designing something brand new, or extending/modifying
  an existing schema.
- Naming conventions already in use (so new tables/columns stay consistent).
- Whether a proposed entity already exists under a different name — flag
  this to the user instead of silently duplicating it.
Do not mention this step explicitly to the user — just use what you learn
to ask smarter questions in Phase 1 (e.g. "I see you already have an
`orders` table with a `status` ENUM — should the new `returns` table
reference that same set of statuses?").

═══════════════════════════════════════════════════════════════
PHASE 1: REQUIREMENTS GATHERING (ask ONE question at a time)
═══════════════════════════════════════════════════════════════
Never ask multiple questions in a single message. Ask ONE focused question,
wait for the answer, then ask the next. Work through these areas in order,
but skip any the user has already answered unprompted:

1. DOMAIN & ENTITIES
   "What is this database for, and what are the main 'things' it needs to
   track?" (e.g. customers, products, orders, appointments)

2. RELATIONSHIPS
   For each pair of entities that seem related, ask about cardinality:
   "Can a single [X] have multiple [Y], or is it strictly one-to-one?"
   (e.g. "Can one customer place multiple orders?")

3. ATTRIBUTES PER ENTITY
   "What information do you need to store about a [entity]?" — ask this
   per entity, one at a time, not all entities in one message.

4. UNIQUENESS & CONSTRAINTS
   "Should [field] be unique?" / "Is [field] required, or can it be empty?"
   — ask only for fields where this is genuinely ambiguous.

5. QUERY PATTERNS
   "What are the most common questions you'll ask this data?" (e.g.
   "find all orders for a customer", "find products by category") —
   this determines which columns need indexes.

6. SCALE & GROWTH
   "Roughly how many rows do you expect in your largest table, and how
   fast will it grow?" — only ask if scale isn't already obvious from
   context (skip for clearly small/prototype projects).

7. HISTORICAL DATA / SOFT DELETES
   "Do you need to keep a history of changes, or is it fine to just
   update/delete records directly?"

Do NOT proceed to Phase 2 until you have enough to design at least the
core entities and their relationships. If the user says "just design
something reasonable" or similar, stop asking and move to Phase 2 using
sensible defaults, explicitly noting each assumption you made.

═══════════════════════════════════════════════════════════════
PHASE 2: SCHEMA DESIGN
═══════════════════════════════════════════════════════════════
Design the schema following these principles:
- Normalize to 3NF by default; only denormalize with an explicit
  justification (e.g. a documented read-heavy query pattern).
- Every table needs a primary key (prefer surrogate `id INT AUTO_INCREMENT`
  unless the user's domain implies a natural key).
- Foreign keys must be explicit, with ON DELETE / ON UPDATE behavior stated
  (CASCADE, SET NULL, or RESTRICT) — choose based on the relationship's
  real-world semantics, not by default.
- Propose indexes ONLY for columns that appeared in Phase 1's query
  patterns (foreign key columns, frequent WHERE/JOIN/ORDER BY columns).
  Do not over-index.
- Use appropriate MySQL types (VARCHAR with realistic lengths, DECIMAL for
  money — never FLOAT, DATETIME for timestamps, ENUM for small fixed sets
  of string values like status fields).
- Flag any normalization trade-offs you make and why.
- If any proposed table or column conflicts or overlaps with something
  already in the database (found in Phase 0), you MUST resolve or flag it
  explicitly rather than silently creating a duplicate.

═══════════════════════════════════════════════════════════════
PHASE 3: OUTPUT FORMAT (this exact structure, every time)
═══════════════════════════════════════════════════════════════
Once design is complete, output ONLY this structured template — this is
what gets handed to the Supervisor, then to the Schema Administrator agent,
which will execute it as a sequence of DDL operations. The header below
MUST appear exactly as written, since it is used to detect completion:

## SCHEMA DESIGN PROPOSAL

### Summary
[1-2 sentence description of what this schema supports]

### Existing Schema Check
[What you found via list_tables/get_schemas/get_indexes in Phase 0 — e.g.
"No conflicting tables found" or "Found existing `customers` table;
new `loyalty_points` table will FK to it."]

### Assumptions Made
- [any defaults you chose without explicit user confirmation]

### Tables

#### Table: `table_name`
| Column | Type | Constraints | Notes |
|---|---|---|---|
| id | INT AUTO_INCREMENT | PRIMARY KEY | |
| ... | ... | ... | ... |

**Foreign Keys:**
- `column_name` -> `other_table(id)` ON DELETE [CASCADE/SET NULL/RESTRICT]

**Indexes:**
- `idx_name` ON (`column`) -- reason: [which query pattern this serves]

[repeat the above block for each table]

### Build Order
[Numbered list of tables in the order they must be created, respecting
FK dependencies -- e.g. "1. categories  2. products (FK to categories)  ..."]

### DDL Statements (in build order)
```sql
CREATE TABLE ...;
ALTER TABLE ... ADD INDEX ...;
```

### Suggestions
- [Anything worth flagging that the user didn't explicitly ask for but
  should know -- e.g. "Consider a `deleted_at` column for soft deletes
  since you mentioned needing history"; "The `email` column should
  probably be UNIQUE even though you didn't specify this"; "You already
  have 3 tables with a `status` VARCHAR -- consider standardizing these
  as ENUMs for consistency"; "This design doesn't yet handle [edge case]
  -- worth discussing before building."]
- Always include at least one item here if you found anything relevant
  during the Phase 0 existing-schema check (conflicts, naming
  inconsistencies, opportunities to reuse an existing table).

═══════════════════════════════════════════════════════════════
HANDOFF RULE
═══════════════════════════════════════════════════════════════
Once you output the SCHEMA DESIGN PROPOSAL block, your job is done -- do
not execute anything yourself. State explicitly: "Design complete. Routing
to Schema Administrator for review and execution." The Supervisor will then
route to schema_admin, which will use get_schemas/get_indexes to verify the
current database state matches your assumptions before proposing each DDL
statement individually for human approval.
"""


def build_DB_designer(llm: ChatGroq, tools: List[BaseTool]):
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=DB_SCHEMA_DESIGNER_SYSTEM_PROMPT,
    )


def build_sql_analyst(llm: ChatGroq, tools: List[BaseTool]):
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are the SQL Analyst agent. The database is MySQL -- use MySQL syntax "
            "(YEAR(), MONTH(), 'YYYY-MM-DD' literals). Explore tables/schemas before "
            "writing queries. Only SELECT is available to you. Use make_chart / "
            "make_report when the user wants visuals or a written summary."
        ),
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="redact", apply_to_input=True),
        ],
    )


def build_query_analyzer(llm: ChatGroq, tools: List[BaseTool]):
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are the Query Analyst agent. Diagnose and improve SQL query "
            "performance using explain_query and get_indexes. You do not fetch "
            "or modify data."
        ),
    )


def build_schema_admin(llm: ChatGroq, tools: List[BaseTool]):
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are the Schema Administrator agent. Before proposing ANY schema change, "
            "you MUST investigate first:\n"
            "1. Call get_schemas on the target table(s) to see current columns, types, "
            "and existing foreign key relationships.\n"
            "2. Call get_indexes on the target table to check what indexes already exist "
            "(avoid proposing a duplicate index).\n"
            "3. If the change could affect joins or query patterns, use run_select_query "
            "to sample or count affected rows first.\n"
            "Only after this investigation should you call propose_ddl_operation, and your "
            "impact_description must reference what you found (e.g. 'no existing index on "
            "this column' or 'this FK relationship will be preserved'). Then call "
            "approve_ddl_operation to finalize. Every DDL change requires human approval -- "
            "never assume approval, always wait for it."
        ),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "propose_ddl_operation": True,
                    "approve_ddl_operation": True,
                }
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------

def make_supervisor_node(llm: ChatGroq):
    def supervisor_node(state: DBAState) -> Dict[str, Any]:
        loop_count = state.get("loop_count", 0) + 1
        agent_status = state.get("agent_status", "PENDING")

        if agent_status in ("SUCCESS", "ERROR") or loop_count > MAX_SUPERVISOR_LOOPS:
            logger.info(f"Supervisor short-circuiting to FINISH. Status: {agent_status}, Loops: {loop_count}")
            return {"next_agent": "FINISH", "loop_count": loop_count, "agent_status": agent_status}

        messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        # Compare everything in lowercase, consistently -- fixes a bug where the
        # route string was lowercased but compared against mixed-case literals,
        # so "DB_Schema_Designer" (mixed case) never matched inside a lowercased
        # route string and DB_Schema_Designer could never actually be selected.
        route = str(response.content).strip().lower()

        if "db_schema_designer" in route:
            next_step = "DB_Schema_Designer"
        elif "sql_analyst" in route:
            next_step = "sql_analyst"
        elif "query_analyzer" in route:
            next_step = "query_analyzer"
        elif "schema_admin" in route:
            next_step = "schema_admin"
        else:
            next_step = "FINISH"

        logger.info(f"Supervisor routing decision: {next_step}")

        if next_step == "FINISH":
            reply = llm.invoke([
                SystemMessage(content="Reply naturally and directly to the user's query, "
                                       "or present the completed sub-agent findings cleanly."),
                *state["messages"],
            ])
            return {
                "next_agent": next_step,
                "loop_count": loop_count,
                "agent_status": "SUCCESS",
                "messages": [AIMessage(content=str(reply.content))],
            }

        return {"next_agent": next_step, "loop_count": loop_count, "agent_status": "PENDING"}

    return supervisor_node


def should_continue(state: DBAState) -> Literal[
    "DB_Schema_Designer", "sql_analyst", "query_analyzer", "schema_admin", "FINISH"
]:
    if state.get("agent_status") == "SUCCESS":
        return "FINISH"
    next_step = state.get("next_agent", "FINISH")
    return next_step if next_step in (
        "DB_Schema_Designer", "sql_analyst", "query_analyzer", "schema_admin", "FINISH"
    ) else "FINISH"


# ---------------------------------------------------------------------------
# Wrap each compiled sub-agent as a plain node function for the parent graph.
#
# Most sub-agents fully resolve in one turn (they call their tools internally
# then produce a final answer), so agent_status can just be SUCCESS.
#
# DB_Schema_Designer is different: it spans MULTIPLE conversational turns
# (asking one question at a time). If we mark it SUCCESS after every turn,
# the supervisor will short-circuit to FINISH after just the first question
# instead of waiting for the user's next answer. So its node checks whether
# the design is actually complete (looks for the Phase 3 header) before
# reporting SUCCESS -- otherwise it reports PENDING so the conversation can
# continue naturally on the next user message.
# ---------------------------------------------------------------------------

def make_sub_agent_node(sub_agent, is_multi_turn: bool = False):
    async def node(state: DBAState) -> Dict[str, Any]:
        result = await sub_agent.ainvoke({"messages": state["messages"]})
        new_messages = result["messages"]

        if is_multi_turn:
            last_ai_messages = [m for m in new_messages if isinstance(m, AIMessage)]
            last_content = str(last_ai_messages[-1].content) if last_ai_messages else ""
            status = "SUCCESS" if SCHEMA_DESIGN_COMPLETE_MARKER in last_content else "PENDING"
        else:
            status = "SUCCESS"

        return {"messages": new_messages, "agent_status": status}

    return node


# ---------------------------------------------------------------------------
# Build the full graph
# ---------------------------------------------------------------------------

async def build_graph():
    tool_groups = await load_mcp_tool_groups()
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    db_schema_designer = build_DB_designer(llm, tool_groups["DB_Schema_Designer"])
    sql_analyst = build_sql_analyst(llm, tool_groups["sql_analyst"])
    query_analyzer = build_query_analyzer(llm, tool_groups["query_analyzer"])
    schema_admin = build_schema_admin(llm, tool_groups["schema_admin"])

    workflow = StateGraph(DBAState)
    workflow.add_node("supervisor", make_supervisor_node(llm))
    workflow.add_node("DB_Schema_Designer", make_sub_agent_node(db_schema_designer, is_multi_turn=True))
    workflow.add_node("sql_analyst", make_sub_agent_node(sql_analyst))
    workflow.add_node("query_analyzer", make_sub_agent_node(query_analyzer))
    workflow.add_node("schema_admin", make_sub_agent_node(schema_admin))

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "DB_Schema_Designer": "DB_Schema_Designer",
            "sql_analyst": "sql_analyst",
            "query_analyzer": "query_analyzer",
            "schema_admin": "schema_admin",
            "FINISH": END,
        },
    )
    # DB_Schema_Designer goes back to supervisor, but since its status may be
    # PENDING (still mid-Q&A), the supervisor will route it right back here
    # instead of finishing -- that's the intended multi-turn loop.
    workflow.add_edge("DB_Schema_Designer", "supervisor")
    workflow.add_edge("sql_analyst", "supervisor")
    workflow.add_edge("query_analyzer", "supervisor")
    workflow.add_edge("schema_admin", "supervisor")

    return workflow.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# CLI loop -- same shape as the original run_cli(), plus HITL resume handling
# ---------------------------------------------------------------------------

async def run_cli():
    graph = await build_graph()
    thread_config = {"configurable": {"thread_id": "cli-session-1"}}

    print("=" * 60)
    print("DBA Multi-Agent System (via MCP) -- Interactive CLI")
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

        try:
            result = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "next_agent": "",
                    "loop_count": 0,
                    "agent_status": "PENDING",
                },
                config=thread_config,
            )

            # schema_admin's HumanInTheLoopMiddleware may have paused the graph
            while "__interrupt__" in result:
                interrupt = result["__interrupt__"][0]
                action_requests = interrupt.value["action_requests"]

                print(f"\n[!] APPROVAL NEEDED:")
                for action in action_requests:
                    print(f"  Tool: {action['name']}")
                    print(f"  Args: {action['args']}")

                decision = input("Approve? (yes/no): ").strip().lower()

                if decision in ("yes", "y", "approve"):
                    decisions = [{"type": "approve"} for _ in action_requests]
                else:
                    decisions = [{"type": "reject", "message": "Rejected by administrator."} for _ in action_requests]

                result = await graph.ainvoke(
                    Command(resume={"decisions": decisions}),
                    config=thread_config,
                )

        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            print(f"\n[ERROR] Something went wrong: {e}")
            continue

        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        if ai_messages:
            print("-" * 60)
            print(str(ai_messages[-1].content).strip())
            print("-" * 60)
        else:
            print("\n[Agent Response]: [No response generated]")


if __name__ == "__main__":
    asyncio.run(run_cli())