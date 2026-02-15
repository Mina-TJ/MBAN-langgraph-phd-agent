PhD Outreach AI Agent (LangGraph)
A multi-agent AI system that researches a professor’s academic profile and generates a personalized PhD interest email with Human-in-the-Loop approval before sending.
Built using LangGraph to demonstrate true agentic workflow design (not sequential scripting).
🎯 Project Overview
This system implements a stateful multi-agent workflow:
START
  ↓
Research Agent
  ↓
Email Composer Agent (LLM)
  ↓
Human-in-the-Loop Approval
  ↓
Gmail Sender Agent
  ↓
END
🧠 What the AI Agent Does
🔍 Research Agent
Scrapes professor’s official university profile
Falls back to Google Scholar if needed
Extracts research interests and publication data
✍️ Email Composer Agent
Uses OpenAI or Anthropic LLM
Generates a concise, publication-aware PhD interest email
Enforces formatting constraints (no placeholders)
✋ Human-in-the-Loop (HITL)
Pauses execution
Requires explicit approval before sending
📤 Email Sender Agent
Sends email securely via Gmail SMTP
Uses environment variables for credentials
🏗️ Architecture Features
LangGraph StateGraph
Shared structured state (TypedDict)
Conditional routing
Checkpoint memory
Tool integration (web scraping + SMTP)
LLM-driven generation
Human approval gate
🛠 Tech Stack
Python
LangGraph
LangChain
OpenAI / Anthropic
BeautifulSoup
Gmail SMTP
dotenv
⚙️ Setup
uv venv
source .venv/bin/activate
uv pip install -e .
python phd_agent.py
🔐 Required Environment Variables
OPENAI_API_KEY=...
# OR
ANTHROPIC_API_KEY=...

SENDER_EMAIL=your_email@gmail.com
APP_PASSWORD=your_gmail_app_password
RECEIVER_EMAIL=test_receiver@email.com
📌 Purpose
This project demonstrates:
AI Agent design
Multi-agent orchestration
Safe LLM automation
Human-controlled execution flow
Designed as an academic assignment to showcase agentic workflow engineering using LangGraph.