"""
PhD Interest Email - Multi-Agent System using LangGraph
Proper agentic workflow using LangGraph (research -> compose -> HITL -> send).
"""

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import TypedDict, Annotated, Literal
import operator

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


# =============================================================================
# LOAD ENV VARS
# =============================================================================
load_dotenv()


# =============================================================================
# STATE DEFINITION
# =============================================================================
class AgentState(TypedDict):
    # Input
    professor_name: str
    scholar_url: str
    university_profile_url: str
    student_background: str
    sender_email: str
    app_password: str
    receiver_email: str

    # Research results
    research_data: dict

    # Email content
    email_subject: str
    email_body: str

    # Workflow control
    messages: Annotated[list, operator.add]
    human_approved: bool
    email_sent: bool
    error_message: str


# =============================================================================
# TOOLS / FUNCTIONS
# =============================================================================
def scrape_university_profile(profile_url: str, professor_name: str) -> dict:
    """Tool: Scrape professor's public university profile page (more reliable than Scholar)."""
    print(f"\n🏫 [TOOL] Scraping university profile for {professor_name}...")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        }
        r = requests.get(profile_url, headers=headers, timeout=15)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # Pull main visible text
        text = soup.get_text("\n", strip=True)

        keywords = [
            "supply chain",
            "logistics",
            "revenue management",
            "pricing",
            "optimization",
            "operations research",
            "network simulation",
            "healthcare",
            "data analytics",
            "business optimization",
        ]
        found = []
        lower = text.lower()
        for k in keywords:
            if k in lower:
                found.append(k.title())

        # Deduplicate while keeping order
        seen = set()
        interests = []
        for x in found:
            if x not in seen:
                interests.append(x)
                seen.add(x)

        # Short snippet (optional, for debug/demo)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        bio_snippet = "\n".join(lines[0:6])

        return {
            "name": professor_name,
            "affiliation": "Saint Mary's University (Sobey School of Business)",
            "interests": interests[:6] if interests else [],
            "publications": [],
            "source": "university_profile",
            "bio_snippet": bio_snippet,
        }

    except Exception as e:
        print(f"  [TOOL] University profile scrape failed: {e}")
        return {}


def scrape_google_scholar(scholar_url: str, professor_name: str) -> dict:
    """Tool: Scrape professor's Google Scholar profile (best-effort, may be blocked)."""
    print(f"\n [TOOL] Scraping Google Scholar for {professor_name}...")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        }
        response = requests.get(scholar_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        research_data = {
            "name": professor_name,
            "affiliation": "",
            "interests": [],
            "publications": [],
        }

        aff = soup.find("div", {"class": "gsc_prf_il"})
        if aff:
            research_data["affiliation"] = aff.get_text(strip=True)

        interest_links = soup.find_all("a", {"class": "gsc_prf_inta"})
        for interest in interest_links[:5]:
            txt = interest.get_text(strip=True)
            if txt:
                research_data["interests"].append(txt)

        pub_rows = soup.find_all("tr", {"class": "gsc_a_tr"})[:5]
        for row in pub_rows:
            title_el = row.find("a", {"class": "gsc_a_at"})
            cites_el = row.find("a", {"class": "gsc_a_ac"})
            if title_el:
                research_data["publications"].append(
                    {
                        "title": title_el.get_text(strip=True),
                        "citations": cites_el.get_text(strip=True) if cites_el else "0",
                    }
                )

        if not research_data["affiliation"]:
            research_data["affiliation"] = "N/A"

        research_data["source"] = "google_scholar"
        print(
            f" [TOOL] Found {len(research_data['interests'])} interests, {len(research_data['publications'])} publications"
        )
        return research_data

    except Exception as e:
        print(f"  [TOOL] Scholar scraping failed, using fallback data: {str(e)}")
        return {
            "name": professor_name,
            "affiliation": "N/A",
            "interests": ["Supply Chain Management", "Revenue Management", "Pricing Strategies", "Operations Research"],
            "publications": [
                {
                    "title": "Game theoretical perspectives on dual-channel supply chain competition",
                    "citations": "579",
                }
            ],
            "source": "fallback",
        }


def send_email_via_gmail(
    sender_email: str,
    app_password: str,
    receiver_email: str,
    subject: str,
    body: str,
) -> dict:
    """Tool: Send email via Gmail SMTP."""
    print(f"\n [TOOL] Sending email to {receiver_email}...")

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)

        print(" [TOOL] Email sent successfully!")
        return {"status": "success", "message": "Email sent successfully"}

    except Exception as e:
        print(f" [TOOL] Failed to send email: {str(e)}")
        return {"status": "error", "message": str(e)}


# =============================================================================
# AGENT NODES
# =============================================================================
def research_agent_node(state: AgentState) -> AgentState:
    print("\n" + "=" * 70)
    print("🔍 RESEARCH AGENT NODE")
    print("=" * 70)

    prof_name = state["professor_name"]

    # 1) Try university profile first
    uni_url = state.get("university_profile_url", "")
    research_data = {}
    if uni_url:
        research_data = scrape_university_profile(uni_url, prof_name)

    # 2) If university scrape failed / empty, try Scholar (may fall back)
    if not research_data or not research_data.get("interests"):
        research_data = scrape_google_scholar(state["scholar_url"], prof_name)
        research_data["source"] = research_data.get("source", "google_scholar_or_fallback")

    state["research_data"] = research_data
    state["messages"].append(
        AIMessage(
            content=(
                f"Research complete from {research_data.get('source','unknown')}. "
                f"Found {len(research_data.get('interests', []))} interest keywords."
            )
        )
    )
    return state


def _pick_llm():
    """Prefer Anthropic if present, else OpenAI; else None."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7, api_key=anthropic_key)

    if openai_key:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_key)

    return None


def email_composer_agent_node(state: AgentState) -> AgentState:
    print("\n" + "=" * 70)
    print("  EMAIL COMPOSER AGENT NODE")
    print("=" * 70)

    research_data = state["research_data"]
    llm = _pick_llm()

    interests = research_data.get("interests", [])
    interests_str = ", ".join(interests) if interests else "N/A"

    top_pub_title = research_data["publications"][0]["title"] if research_data.get("publications") else "N/A"
    top_pub_cites = research_data["publications"][0]["citations"] if research_data.get("publications") else "0"

    prompt = f"""
You are writing a short, high-quality PhD interest email.

Student name: Mina Tavakkoli Jouybari.
Hard rules:
- Do NOT use placeholders like [Your Name] or bracket text.
- Email must be concise: 140–190 words total.
- Use 3 short paragraphs + a single-line closing.
- Include ONE clear call-to-action: request a 15–20 minute meeting.
- Mention exactly ONE publication title (if provided).
- Avoid repetition and generic filler.
- Output must follow the exact format below.

Professor information:
- Name: {research_data['name']}
- Affiliation: {research_data.get('affiliation', 'N/A')}
- Research Interests: {interests_str}
- Publication to reference: {top_pub_title} ({top_pub_cites} citations)

Student background:
{state['student_background']}

Output format (exactly):
SUBJECT: <one line>
---
BODY:
<email body with no SUBJECT line inside>
"""

    if llm is None:
        subject = "Prospective PhD Student – Interest in Supply Chain & Revenue Management"
        body = f"""Dear Professor {research_data['name']},

I am Mina Tavakkoli Jouybari, currently completing my Master's degree in Business Analytics. My thesis focuses on supply chain optimization using machine learning techniques, which has equipped me with strong skills in optimization, statistical modeling, and data analysis.

I am particularly interested in your work in {', '.join(interests[:2]) if interests else 'your research area'}, especially your publication “{top_pub_title}.” I believe my background aligns well with your research interests and I am eager to explore potential PhD opportunities in your group.

Could we schedule a 15–20 minute meeting to discuss how my interests might align with your ongoing projects?

Best regards,  
Mina Tavakkoli Jouybari
"""
        state["email_subject"] = subject
        state["email_body"] = body
        state["messages"].append(AIMessage(content=f"Email composed (fallback) with subject: {subject[:50]}..."))
        print(" Email composed via fallback")
        return state

    messages = [
        SystemMessage(content="You write concise, professor-ready academic emails."),
        HumanMessage(content=prompt),
    ]

    print("🤖 LLM composing email...")
    response = llm.invoke(messages)
    email_text = response.content if isinstance(response.content, str) else str(response.content)

    if "SUBJECT:" in email_text and "BODY:" in email_text:
        parts = email_text.split("---", 1)
        subject_part = parts[0].replace("SUBJECT:", "").strip()
        body_part = parts[1].split("BODY:", 1)[-1].strip()
    else:
        subject_part = "Prospective PhD Student – Interest in Supply Chain & Revenue Management"
        body_part = email_text.strip()

    # Remove any accidental SUBJECT lines inside body
    body_lines = []
    for line in body_part.splitlines():
        if line.strip().lower().startswith("subject:"):
            continue
        if line.strip() == "---":
            continue
        body_lines.append(line)
    body_part = "\n".join(body_lines).strip()

    # Safety: reject placeholders
    banned_tokens = ["[Your Name]", "[Your Full Name]", "[Contact Information]", "["]
    if any(tok in body_part for tok in banned_tokens):
        subject_part = "Prospective PhD Student – Interest in Supply Chain & Revenue Management"
        body_part = f"""Dear Professor {research_data['name']},

I am Mina Tavakkoli Jouybari, currently completing my Master's degree in Business Analytics. My thesis focuses on supply chain optimization using machine learning techniques, which has equipped me with strong skills in optimization, statistical modeling, and data analysis.

I am particularly interested in your work in {', '.join(interests[:2]) if interests else 'your research area'}, especially your publication “{top_pub_title}.” I believe my background aligns well with your research interests and I am eager to explore potential PhD opportunities in your group.

Could we schedule a 15–20 minute meeting to discuss how my interests might align with your ongoing projects?

Best regards,  
Mina Tavakkoli Jouybari
"""

    state["email_subject"] = subject_part
    state["email_body"] = body_part
    state["messages"].append(AIMessage(content=f"Email composed with subject: {subject_part[:50]}..."))
    print(" Email composed (concise professor style)")
    return state


def human_approval_node(state: AgentState) -> AgentState:
    print("\n" + "=" * 70)
    print(" HUMAN-IN-THE-LOOP APPROVAL NODE")
    print("=" * 70)

    print("\n📧 EMAIL PREVIEW:")
    print(f"\nSUBJECT: {state['email_subject']}")
    print("\n" + "-" * 70)
    print(state["email_body"])
    print("-" * 70)

    auto_approve = os.getenv("AUTO_APPROVE", "").lower() in {"1", "true", "yes", "y"}
    if auto_approve:
        state["human_approved"] = True
        state["messages"].append(HumanMessage(content="Email auto-approved (AUTO_APPROVE=true)"))
        print(" Email auto-approved!")
        return state

    while True:
        decision = input("\n✋ Do you approve sending this email? (yes/no): ").strip().lower()
        if decision in ["yes", "y"]:
            state["human_approved"] = True
            state["messages"].append(HumanMessage(content="Email approved by human"))
            print(" Email approved!")
            break
        if decision in ["no", "n"]:
            state["human_approved"] = False
            state["messages"].append(HumanMessage(content="Email rejected by human"))
            print(" Email rejected.")
            break
        print("Please enter 'yes' or 'no'")

    return state


def email_sender_node(state: AgentState) -> AgentState:
    print("\n" + "=" * 70)
    print(" EMAIL SENDER NODE")
    print("=" * 70)

    result = send_email_via_gmail(
        state["sender_email"],
        state["app_password"],
        state["receiver_email"],
        state["email_subject"],
        state["email_body"],
    )

    if result["status"] == "success":
        state["email_sent"] = True
        state["messages"].append(AIMessage(content="Email sent successfully!"))
    else:
        state["email_sent"] = False
        state["error_message"] = result["message"]
        state["messages"].append(AIMessage(content=f"Email sending failed: {result['message']}"))

    return state


def should_send_email(state: AgentState) -> Literal["send_email", "end"]:
    return "send_email" if state["human_approved"] else "end"


# =============================================================================
# BUILD THE LANGGRAPH WORKFLOW
# =============================================================================
def create_phd_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("email_composer_agent", email_composer_agent_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("email_sender", email_sender_node)

    workflow.add_edge(START, "research_agent")
    workflow.add_edge("research_agent", "email_composer_agent")
    workflow.add_edge("email_composer_agent", "human_approval")

    workflow.add_conditional_edges(
        "human_approval",
        should_send_email,
        {"send_email": "email_sender", "end": END},
    )

    workflow.add_edge("email_sender", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print(
        """
╔═══════════════════════════════════════════════════════════════╗
║    PhD INTEREST EMAIL - LANGGRAPH MULTI-AGENT SYSTEM         ║
║                                                               ║
║  Graph: START → Research → Compose (LLM) → HITL → Send → END  ║
╚═══════════════════════════════════════════════════════════════╝
"""
    )

    sender_email = os.getenv("SENDER_EMAIL", "")
    app_password = os.getenv("APP_PASSWORD", "")
    receiver_email = os.getenv("RECEIVER_EMAIL", "")

    if not sender_email or not app_password:
        print("  ERROR: Missing SENDER_EMAIL or APP_PASSWORD in .env")
        return

    if not receiver_email:
        print("  ERROR: Missing RECEIVER_EMAIL in .env")
        return

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n  WARNING: No LLM API key found!")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env to use an LLM.")
        print("Proceeding with fallback email generation.\n")

    initial_state: AgentState = {
        "professor_name": "Michael Zhang",
        "scholar_url": "https://scholar.google.ca/citations?user=5k8zF1kAAAAJ",
        "university_profile_url": "https://sobeyschoolprosperitynetwork.com/dr-michael-zhang",
        "student_background": """I am currently completing my Master's degree in Business Analytics,
where I have developed strong skills in optimization, statistical modeling, and data analysis.
My thesis focuses on supply chain optimization using machine learning techniques, which has
given me hands-on experience with both theoretical and practical aspects of operations research.""",
        "sender_email": sender_email,
        "app_password": app_password,
        "receiver_email": receiver_email,
        "research_data": {},
        "email_subject": "",
        "email_body": "",
        "messages": [],
        "human_approved": False,
        "email_sent": False,
        "error_message": "",
    }

    print("\n🔧 Building LangGraph workflow...")
    app = create_phd_agent_graph()
    print(" Graph compiled successfully!")
    print("\n Starting agentic workflow execution...\n")

    config = {"configurable": {"thread_id": "phd_email_001"}}

    try:
        final_state = app.invoke(initial_state, config)

        print("\n" + "=" * 70)
        print("🎉 WORKFLOW EXECUTION COMPLETE")
        print("=" * 70)
        print(f"\n Research completed: {final_state['research_data'].get('name', 'N/A')}")
        print(f" Email composed: {len(final_state['email_body'])} characters")
        print(f" Human approval: {'Approved' if final_state['human_approved'] else 'Rejected'}")
        print(f" Email sent: {'Yes' if final_state['email_sent'] else 'No'}")

        if final_state["error_message"]:
            print(f"\n  Error: {final_state['error_message']}")

    except Exception as e:
        print(f"\n Error during execution: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
