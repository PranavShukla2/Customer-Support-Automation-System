from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from state import SupportState
from router import classify_intent, route_query, route_after_department
from agents import sales_agent, technical_agent, billing_agent, account_agent
from memory import checkpointer, memory_agent
from supervisor import human_review_node, supervisor_agent

builder = StateGraph(SupportState)

builder.add_node("classifier", classify_intent)
builder.add_node("sales", sales_agent)
builder.add_node("technical", technical_agent)
builder.add_node("billing", billing_agent)
builder.add_node("account", account_agent)
builder.add_node("memory", memory_agent)
builder.add_node("human_review", human_review_node)
builder.add_node("supervisor", supervisor_agent)

builder.set_entry_point("classifier")

builder.add_conditional_edges(
    "classifier",
    route_query,
    {
        "sales": "sales",
        "technical": "technical",
        "billing": "billing",
        "account": "account",
        "memory": "memory",
    },
)

for dept in ["sales", "technical", "billing", "account", "memory"]:
    builder.add_conditional_edges(
        dept,
        route_after_department,
        {
            "human_review": "human_review",
            "supervisor": "supervisor",
        },
    )

builder.add_edge("human_review", "supervisor")
builder.add_edge("supervisor", END)

app = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"],
)

DEMO_QUERIES = [
    "What pricing plans do you offer?",
    "My API calls are timing out on the batch endpoint.",
    "How do I reset my password?",
    "I'd like a refund for last month's charge.",
    "What was my previous issue?",
]

config = {"configurable": {"thread_id": "test_session"}}

def run_demo():
    # Note: Free-tier Gemini API users might need to add a delay (e.g., time.sleep(15)) between queries to avoid rate limits.
    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print("=" * 60)

        result = app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )

        snapshot = app.get_state(config)
        if snapshot.next:
            print(f"\n⏸  Graph interrupted before: {snapshot.next}")
            print(f"   Draft response: {snapshot.values.get('draft_response', '')[:120]}...")
            print(f"   requires_approval: {snapshot.values.get('requires_approval')}")

            app.update_state(config, {"is_approved": True}, as_node="human_review")
            print("   ✅ Human approved — resuming execution...")

            result = app.invoke(None, config=config)

        final = result.get("final_response", "(no final response)")
        print(f"\n✉  Final Response:\n{final}")

if __name__ == "__main__":
    run_demo()