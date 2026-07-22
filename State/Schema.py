from langgraph.graph import MessagesState

class DBAState(MessagesState):
    next_agent: str
    loop_count: int
    agent_status: str