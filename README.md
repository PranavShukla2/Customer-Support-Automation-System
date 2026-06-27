# Customer Support Automation System

A multi-agent customer support automation system built using [LangChain](https://python.langchain.com/) and [LangGraph](https://python.langchain.com/docs/langgraph/). This system intelligently routes user queries to specialized department agents, utilizes Retrieval-Augmented Generation (RAG) for context-aware responses, maintains conversation history, and includes a human-in-the-loop mechanism for sensitive operations.

Powered by the **Google Gemini API** (`gemini-2.5-flash`), this system delivers fast, professional, and highly accurate customer support.

---

## Architecture Overview

The system models the customer support flow as a state machine (`StateGraph`) using LangGraph. The core components are:

- **Intent Classifier (`router.py`)**: Uses Gemini's structured output capabilities to analyze the user's query and route it to exactly one of five destinations (Sales, Technical, Billing, Account, or Memory).
- **Specialized Agents (`agents.py`)**:
  - **Sales Agent**: Handles pricing, trials, and plan information.
  - **Technical Agent**: Addresses API limits, webhooks, and integrations.
  - **Billing Agent**: Manages refunds, downgrades, and cancellations. (Automatically triggers human review for sensitive keywords).
  - **Account Agent**: Deals with password resets, MFA, and account deletion.
- **Memory Agent (`memory.py`)**: Uses an SQLite checkpointer to retrieve past conversational history when customers ask about previous interactions.
- **Human Review Node (`supervisor.py`)**: An interrupt node that halts the graph execution for sensitive queries (e.g., refunds) until a human supervisor approves the draft response.
- **Supervisor Agent (`supervisor.py`)**: The final node in the graph. It takes the drafted response from any department and refines it into a polished, warm, and professional customer-facing reply, ensuring internal context and system logs are stripped out.

---

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Dependencies
Clone the repository and install the required packages:

```bash
git clone https://github.com/PranavShukla2/Customer-Support-Automation-System.git
cd Customer-Support-Automation-System
pip install -r requirements.txt
```

### 3. Configure the Google Gemini API Key
The system relies on the Google Gemini API. You must obtain an API key from Google AI Studio and export it to your environment variables:

```bash
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

### 4. Run the Demo
A built-in demo script is provided to showcase the end-to-end pipeline, including intent routing, RAG retrieval, human-in-the-loop interrupts, and supervisor polishing.

```bash
python main.py
```

> **Note for Free-Tier Gemini API Users:** 
> The free tier of the Gemini API has strict rate limits. The script `main.py` makes multiple sequential LLM calls per query (Classifier → Agent → Supervisor). If you encounter a `429 RESOURCE_EXHAUSTED` error, you can uncomment or add a `time.sleep(15)` delay between the queries in the demo loop.

---

## File Structure

- `main.py`: Compiles the LangGraph state machine and runs the demo.
- `router.py`: Contains the `classify_intent` node and conditional routing logic.
- `agents.py`: Contains the RAG knowledge base and the specialized department agents.
- `supervisor.py`: Houses the `human_review` interrupt node and the `supervisor_agent` polishing logic.
- `memory.py`: Configures the SQLite checkpointer and the conversational memory agent.
- `state.py`: Defines the `SupportState` TypedDict used to pass state through the graph.
- `requirements.txt`: Project dependencies including `langgraph`, `langchain-google-genai`, and `pydantic`.
- `support_memory.db`: (Auto-generated) SQLite database file used for thread persistence.
