import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import SupportState


DB_PATH = "support_memory.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def memory_agent(state: SupportState) -> dict:
    query = state["messages"][-1].content

    history_lines = []
    for msg in state["messages"][:-1]:
        role = "Customer" if msg.type == "human" else "Agent"
        history_lines.append(f"{role}: {msg.content}")

    history_text = "\n".join(history_lines) if history_lines else "No previous conversation history."

    response = llm.invoke([
        SystemMessage(content=(
            "You are a support memory assistant. The customer is asking about their "
            "previous interactions. Use the conversation history below to answer their "
            "question accurately. If no relevant history exists, say so honestly.\n\n"
            f"Conversation History:\n{history_text}"
        )),
        HumanMessage(content=query),
    ])

    return {
        "draft_response": response.content,
        "requires_approval": False,
    }
