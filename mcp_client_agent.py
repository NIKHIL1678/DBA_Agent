import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq

load_dotenv()

async def build_agent():
    client = MultiServerMCPClient({
        "dba_agent": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8000/mcp",
        }
    })
    tools = await client.get_tools()

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    agent = create_agent(
        model=llm,
        tools=tools,
        middleware=[
            # Redacts PII from user input before it reaches the LLM —
            # replaces your Middlewares/PII_filters.py
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="redact", apply_to_input=True),

            # Pauses for human approval before running risky tools —
            # replaces your Middlewares/HIL_Middleware.py + blocking input()
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "propose_ddl_operation": True,
                    "approve_ddl_operation": True,
                }
            ),
        ],
        checkpointer=InMemorySaver(),  # gives you multi-turn memory, replaces manual conversation_state
    )
    return agent


async def run_cli():
    agent = await build_agent()
    thread_config = {"configurable": {"thread_id": "cli-session-1"}}

    print("=" * 60)
    print("DBA MCP Agent — Interactive CLI (with memory + PII + HIL)")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        user_input = input("\n[You]: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=thread_config,
        )

        # If HumanInTheLoopMiddleware paused execution, result will contain
        # an interrupt instead of a normal final answer — handle that case:
        if "__interrupt__" in result:
            interrupt = result["__interrupt__"][0]
            print(f"\n⚠️  APPROVAL NEEDED: {interrupt.value}")
            decision = input("Approve? (yes/no): ").strip().lower()
            # Resume the graph with the human's decision
            from langgraph.types import Command
            result = await agent.ainvoke(
                Command(resume={"decision": decision}),
                config=thread_config,
            )

        print("-" * 60)
        print(result["messages"][-1].content)
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(run_cli())