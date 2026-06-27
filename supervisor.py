from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from state import SupportState


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def human_review_node(state: SupportState) -> dict:
    return {
        "is_approved": True,
    }


def supervisor_agent(state: SupportState) -> dict:
    draft = state.get("draft_response", "")
    query = state["messages"][-1].content

    polished = llm.invoke([
        SystemMessage(content=(
            "You are a senior support supervisor. Your job is to refine the draft "
            "response below into a polished, professional reply.\n\n"
            "Rules:\n"
            "- Address the customer's question directly.\n"
            "- Use a warm, helpful tone.\n"
            "- Never expose internal system logs, tool names, or debug info.\n"
            "- Keep the response concise but complete.\n\n"
            f"Customer query: {query}\n\n"
            f"Draft response:\n{draft}"
        )),
        HumanMessage(content="Please produce the final customer-facing response."),
    ])

    return {
        "final_response": polished.content,
        "messages": [AIMessage(content=polished.content)],
    }
