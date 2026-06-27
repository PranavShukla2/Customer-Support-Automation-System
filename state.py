import operator
from typing import Annotated, Literal, Sequence

from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class SupportState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    customer_id: str
    intent: Literal["Sales", "Technical", "Billing", "Account", "Memory"]
    context: str  # RAG-retrieved data
    requires_approval: bool
    is_approved: bool
    draft_response: str
    final_response: str
