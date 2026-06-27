from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from state import SupportState


class IntentClassification(BaseModel):
    intent: Literal["Sales", "Technical", "Billing", "Account", "Memory"] = Field(
        description="The classified intent of the customer's message."
    )


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
classifier = llm.with_structured_output(IntentClassification)


def classify_intent(state: SupportState) -> dict:
    last_message = state["messages"][-1]
    result = classifier.invoke([last_message])
    return {"intent": result.intent}


def route_query(state: SupportState) -> str:
    return state["intent"].lower()


def route_after_department(state: SupportState) -> str:
    if state["requires_approval"]:
        return "human_review"
    return "supervisor"
