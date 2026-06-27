from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import SupportState


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

RAG_SOURCES = {
    "Sales": "Pricing Guide: Starter plan $29/mo, Pro $79/mo, Enterprise custom pricing. "
             "All plans include 14-day free trial. Volume discounts available for 10+ seats.",
    "Technical": "Technical Manual: Supported integrations — REST API, Webhooks, OAuth2. "
                 "Rate limit: 1000 req/min. Known issue: intermittent timeout on batch endpoints (v2.3.1).",
    "Billing": "Billing Policy: Refunds within 30 days of charge. Pro-rated credits for downgrades. "
               "Cancellations take effect at end of billing cycle. Disputes resolved within 5 business days.",
    "Account": "Account Docs: Password reset via email link (expires in 1 hr). MFA supports TOTP and SMS. "
               "Account deletion requires identity verification and has a 30-day grace period.",
}

APPROVAL_KEYWORDS = {"refund", "cancel", "close", "compensation", "escalate"}


def _generate_draft(query: str, context: str, department: str) -> str:
    response = llm.invoke([
        SystemMessage(content=(
            f"You are a {department} support specialist. Use the provided context to "
            f"write a helpful, concise response to the customer's query.\n\n"
            f"Context:\n{context}"
        )),
        HumanMessage(content=query),
    ])
    return response.content


def sales_agent(state: SupportState) -> dict:
    print("[LOG] Sales Agent invoked.")
    print("[LOG] Retrieving RAG context from company documents...")
    query = state["messages"][-1].content
    context = state.get("context", "") + "\n" + RAG_SOURCES["Sales"]
    draft = _generate_draft(query, context, "Sales")
    return {"context": context, "draft_response": draft, "requires_approval": False}


def technical_agent(state: SupportState) -> dict:
    print("[LOG] Technical Agent invoked.")
    print("[LOG] Retrieving RAG context from company documents...")
    query = state["messages"][-1].content
    context = state.get("context", "") + "\n" + RAG_SOURCES["Technical"]
    draft = _generate_draft(query, context, "Technical")
    return {"context": context, "draft_response": draft, "requires_approval": False}


def billing_agent(state: SupportState) -> dict:
    print("[LOG] Billing Agent invoked.")
    print("[LOG] Retrieving RAG context from company documents...")
    query = state["messages"][-1].content
    context = state.get("context", "") + "\n" + RAG_SOURCES["Billing"]

    query_lower = query.lower()
    needs_approval = any(kw in query_lower for kw in APPROVAL_KEYWORDS)

    draft = _generate_draft(query, context, "Billing")
    return {"context": context, "draft_response": draft, "requires_approval": needs_approval}


def account_agent(state: SupportState) -> dict:
    print("[LOG] Account Agent invoked.")
    print("[LOG] Retrieving RAG context from company documents...")
    query = state["messages"][-1].content
    context = state.get("context", "") + "\n" + RAG_SOURCES["Account"]
    draft = _generate_draft(query, context, "Account")
    return {"context": context, "draft_response": draft, "requires_approval": False}
