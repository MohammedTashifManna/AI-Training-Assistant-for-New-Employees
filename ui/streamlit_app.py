"""Streamlit chatbot interface for the AI Training Assistant.

Run from the project root::

    streamlit run ui/streamlit_app.py

Beyond being a chat window, the UI is the project's main explainability
surface. Every answer shows which of the four paths handled it, which document
sections were retrieved and how strongly they matched, and what the router did.
An onboarding assistant that cannot show its sources is not much better than a
rumour, so the citations are part of the product rather than a debug view.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make ``src/`` importable when launched via ``streamlit run``.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from training_assistant.assistant import AssistantAnswer, TrainingAssistant  # noqa: E402
from training_assistant.config import get_settings  # noqa: E402
from training_assistant.logging_setup import quiet_third_party  # noqa: E402

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Manna's Training Assistant",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="expanded",
)

quiet_third_party()

#: Visual identity per route. Emoji are paired with text labels rather than used
#: alone, so the route is still conveyed to a screen reader or in monochrome.
ROUTE_STYLE: dict[str, tuple[str, str]] = {
    "company": ("🏢", "Company knowledge base"),
    "role": ("👥", "Role & team documents"),
    "admin": ("📋", "Policy & admin documents"),
    "general": ("💬", "Direct answer (no retrieval)"),
}

SAMPLE_QUESTIONS: dict[str, list[str]] = {
    "🏢 Company": [
        "What does Manna's Analytics actually do?",
        "Where are our offices located?",
        "What does AR mean in a calendar invite here?",
    ],
    "👥 Role & team": [
        "When do new engineers join the on-call rotation?",
        "How does the performance rating scale work?",
        "Who owns the customer relationship after a contract is signed?",
    ],
    "📋 Policy & admin": [
        "How many days of annual leave do I get?",
        "What is the hotel cap per night for a work trip to Mumbai?",
        "I think I clicked a phishing link. What do I do?",
    ],
    "💬 General": [
        "Who are you and what can you help me with?",
        "Can you explain what retrieval augmented generation means?",
    ],
}


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def _load_settings():
    """Settings are process-wide and cheap; cache across reruns."""
    return get_settings()


def _init_state() -> None:
    if "assistant" not in st.session_state:
        st.session_state.assistant = TrainingAssistant(_load_settings())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None


def _reset_conversation() -> None:
    st.session_state.assistant = TrainingAssistant(_load_settings())
    st.session_state.messages = []
    st.session_state.pending = None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_route_badge(answer: AssistantAnswer) -> None:
    icon, label = ROUTE_STYLE.get(answer.route, ("❓", answer.route))
    agreement = (
        ""
        if answer.route_agreed_with_rules
        else f" · rule-based classifier suggested **{answer.rule_based_route}**"
    )
    st.caption(f"{icon} Routed to **{label}**{agreement}")


def _render_sources(answer: AssistantAnswer) -> None:
    if not answer.sources:
        if answer.route != "general":
            st.caption("No source passages were retrieved for this answer.")
        return

    with st.expander(f"📄 Sources ({len(answer.sources)} passages retrieved)"):
        st.caption(
            f"Searched **{ROUTE_STYLE.get(answer.route, ('', answer.route))[1]}** "
            f"for: `{answer.retrieval_query or answer.question}`"
        )
        for source in answer.sources:
            st.markdown(f"**[{source.number}] {source.citation}**")
            st.progress(
                min(1.0, max(0.0, source.score)),
                text=f"relevance {source.score:.3f}",
            )
            if source.text:
                snippet = source.text if len(source.text) <= 1200 else source.text[:1200] + " …"
                st.code(snippet, language="markdown", wrap_lines=True)
            if source.source_path:
                st.caption(f"file: `{source.source_path}`")
            st.divider()


def _render_trace(answer: AssistantAnswer) -> None:
    with st.expander("🔍 Routing trace"):
        left, right = st.columns(2)
        with left:
            st.markdown("**Decision**")
            st.markdown(
                f"- Route: `{answer.route}`\n"
                f"- Agent: `{answer.agent or 'n/a'}`\n"
                f"- Rule-based route: `{answer.rule_based_route}` "
                f"(confidence {answer.rule_based_confidence:.2f})\n"
                f"- Agreement: `{answer.route_agreed_with_rules}`"
            )
        with right:
            st.markdown("**Execution**")
            timings = answer.timings
            st.markdown(
                f"- Mode: `{answer.mode}`\n"
                f"- Model turns: `{answer.llm_calls}`\n"
                f"- Tool calls: `{', '.join(answer.tool_calls) or 'none'}`\n"
                f"- Elapsed: `{answer.elapsed_seconds:.2f}s`"
            )
            st.markdown(
                f"- Routing: `{timings.routing_seconds * 1000:.0f}ms`\n"
                f"- Retrieval: `{timings.retrieval_seconds * 1000:.0f}ms` "
                f"(search `{timings.search_seconds * 1000:.0f}ms`)\n"
                f"- Generation: `{timings.generation_seconds * 1000:.0f}ms`\n"
                f"- Unattributed: `{timings.overhead_seconds * 1000:.0f}ms`"
            )
        if answer.transfer_reason:
            st.markdown(f"**Why this route:** {answer.transfer_reason}")
        if answer.error:
            st.error(f"Error: {answer.error}")


def _render_message(message: dict) -> None:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
        return

    answer: AssistantAnswer = message["answer"]
    with st.chat_message("assistant"):
        _render_route_badge(answer)
        st.markdown(answer.answer)
        _render_sources(answer)
        _render_trace(answer)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def _render_sidebar(settings) -> None:
    with st.sidebar:
        st.subheader("Manna's Training Assistant")
        st.caption("Onboarding Q&A over the company knowledge base.")

        if settings.offline:
            st.warning(
                "**Offline mode** — no Gemini API key found.\n\n"
                "Routing and retrieval are fully live, but answers are "
                "*extracted* from the source passages rather than written by a "
                "language model. Add `GOOGLE_API_KEY` to `.env` for "
                "Gemini-generated answers.",
                icon="⚠️",
            )
        else:
            st.success(f"**Live mode** — answers generated by `{settings.model}`.", icon="✅")

        st.divider()
        st.markdown("**Configuration**")
        st.markdown(
            f"- Mode: `{settings.mode}`\n"
            f"- Embeddings: `{settings.embedding_backend}`\n"
            f"- Passages per search: `{settings.top_k}`\n"
            f"- Semantic / lexical mix: `{settings.semantic_weight:.2f}` / "
            f"`{settings.lexical_weight:.2f}`\n"
            f"- Relevance floor: `{settings.min_score}`"
        )

        try:
            from training_assistant.knowledge.retriever import get_retriever

            stats = get_retriever().store.stats()
            st.markdown("**Knowledge index**")
            st.markdown(
                "\n".join(
                    f"- {ROUTE_STYLE.get(domain, ('', domain))[1]}: `{count}` passages"
                    for domain, count in stats.items()
                )
            )
        except FileNotFoundError:
            st.error(
                "No knowledge index found. Run:\n\n"
                "`python scripts/build_index.py`",
                icon="🚫",
            )
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Index could not be loaded: {exc}")

        st.divider()
        st.markdown("**How routing works**")
        st.markdown(
            "A root router agent classifies each question and delegates it to "
            "exactly one specialist:"
        )
        for route, (icon, label) in ROUTE_STYLE.items():
            st.markdown(f"- {icon} `{route}` → {label}")

        st.divider()
        if st.button("🔄 New conversation", use_container_width=True):
            _reset_conversation()
            st.rerun()

        st.caption(
            "Manna's Analytics is a fictional company. All documents in this "
            "knowledge base are synthetic."
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    settings = _load_settings()
    _init_state()
    _render_sidebar(settings)

    st.title("🧭 AI Training Assistant")
    st.markdown(
        "Ask anything about working at **Manna's Analytics**. Your question is "
        "routed to the company knowledge base, the role and team documents, the "
        "policy and admin documents, or answered directly, and every answer "
        "shows its sources."
    )

    if not settings.index_path.is_file():
        st.error(
            f"**No knowledge index at** `{settings.index_path.name}`\n\n"
            "Build it before asking questions:\n\n"
            "```\npython scripts/build_index.py\n```",
            icon="🚫",
        )
        st.stop()

    # Sample questions, shown only at the start of a conversation.
    if not st.session_state.messages:
        st.markdown("**Try one of these**")
        for group, questions in SAMPLE_QUESTIONS.items():
            st.caption(group)
            columns = st.columns(len(questions))
            for column, question in zip(columns, questions):
                with column:
                    if st.button(
                        question,
                        key=f"sample-{question}",
                        use_container_width=True,
                    ):
                        st.session_state.pending = question
                        st.rerun()
        st.divider()

    for message in st.session_state.messages:
        _render_message(message)

    typed = st.chat_input("Ask about policies, your role, or the company…")
    question = typed or st.session_state.pending
    st.session_state.pending = None

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing your question…"):
            answer = st.session_state.assistant.ask(question)
        _render_route_badge(answer)
        st.markdown(answer.answer)
        _render_sources(answer)
        _render_trace(answer)

    st.session_state.messages.append({"role": "assistant", "answer": answer})


main()
