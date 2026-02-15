PhD Outreach AI Agent (LangGraph)
This project is a multi-agent AI system that researches a professor’s academic profile and generates a personalized PhD interest email — but only sends it after human approval.
I built this using LangGraph to demonstrate true agentic workflow design, not just sequential scripting.
💡 What This Project Does
The system runs as a structured AI workflow:
Research → Draft → Pause for Approval → Send
Rather than writing a script that executes top-to-bottom, I designed separate agents that share state, use tools, and make decisions throughout the process.
🧠 How It Works
🔍 Research Agent
Scrapes the professor’s official university page
Falls back to Google Scholar if needed
Extracts research themes and publication data
✍️ Email Composer Agent
Uses an LLM (OpenAI or Anthropic)
Generates a concise, publication-aware PhD interest email
Enforces formatting constraints automatically
✋ Human-in-the-Loop
Pauses execution before sending
Requires explicit approval
📤 Email Sender Agent
Sends the email securely via Gmail SMTP
Manages credentials through environment variables and Gmail App Password authentication
🏗 Why This Is Different
This is not just a script that calls an API.
It uses:
LangGraph StateGraph
Shared structured state (TypedDict)
Conditional routing
Tool integration (web scraping + SMTP)
Memory checkpointing
A human approval gate for safe automation
This architecture makes the system modular, safe, and extensible.
🛠 Tech Stack
Python • LangGraph • LangChain • OpenAI / Anthropic
BeautifulSoup • Gmail SMTP • dotenv
🎓 Context
Built as part of the MBAN program at Saint Mary’s University, this project focuses on designing AI agents that combine LLM reasoning, external tools, and controlled automation.
