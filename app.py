"""UGA Student Resource Chatbot — Streamlit front end.

All retrieval and triage logic lives in resources.py so it can be tested and
measured without a browser (test_resources.py, evaluate.py). This file is
only the interface.

Three deliberate behaviours:

* Provider-agnostic. Groq, Google, and OpenAI all speak the OpenAI wire
  format, so any of their keys works; free providers are preferred.
* It works with no key at all, answering straight from the index, so a public
  demo is useful to anyone who opens it rather than gated behind billing.
* Questions suggesting immediate risk short-circuit to crisis resources
  before anything else runs.
"""

import os

import streamlit as st
from dotenv import load_dotenv

import providers
import resources as rx

load_dotenv()

st.set_page_config(
    page_title="UGA Student Resource Chatbot",
    page_icon="🐾",
    layout="centered",
)


def secret(name):
    """Streamlit Cloud injects st.secrets; local runs use .env.

    Reading st.secrets raises when no secrets file exists, which is the normal
    case on a laptop, so the lookup is guarded rather than assumed.
    """
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


@st.cache_resource
def get_index():
    return rx.build_index()


def stream_answer(client, model, question, matches, history, crisis):
    """Yield the model's reply token by token for st.write_stream."""
    system = rx.CRISIS_SYSTEM_PROMPT if crisis else rx.SYSTEM_PROMPT
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": rx.build_prompt(question, matches, history)},
        ],
        max_tokens=350,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
try:
    index = get_index()
except FileNotFoundError:
    st.error(f"Could not find {rx.DATA_FILE} next to app.py.")
    st.stop()

provider, api_key = providers.resolve(secret)
client = None
model = None
if provider:
    try:
        client = providers.make_client(provider, api_key)
        model = provider.model(secret)
    except Exception as exc:
        st.warning(f"Could not start the {provider.name} client, using search only. ({exc})")
        provider = None

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Browse the index")
    st.caption(f"{len(index)} resources across {len(index.categories)} categories")

    chosen = st.selectbox("Category", ["All"] + index.categories)
    view = index.frame if chosen == "All" else index.frame[index.frame["category"] == chosen]
    st.dataframe(
        view[["resource_name", "description"]],
        hide_index=True,
        use_container_width=True,
        height=320,
    )

    if st.session_state.messages and st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "A student-maintained index, not an official UGA directory. "
        "Confirm details on each department's own page."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("UGA Student Resource Chatbot")
st.caption(
    "Ask about advising, tutoring, health, money, housing, or campus life. "
    f"Answers come from an index of {len(index)} UGA resources — never invented."
)

if client is None:
    free_names = " or ".join(p.name for p in providers.FREE_PROVIDERS)
    st.info(
        "**Search mode** — no model key is configured, so replies list matching "
        "resources directly instead of being written by a model. Retrieval, "
        f"citations and crisis triage all work the same. Add a free {free_names} "
        "key to turn on written answers.",
        icon="🔍",
    )
else:
    st.caption(
        f"Answers written by **{provider.label}**"
        + (" · free tier" if provider.free else "")
    )

SUGGESTIONS = [
    "Where can I get free tutoring?",
    "I think I'm depressed",
    "How do I find an internship?",
    "I can't afford groceries this month",
]

if not st.session_state.messages:
    st.write("**Try one of these:**")
    for row_start in (0, 2):
        for col, suggestion in zip(st.columns(2), SUGGESTIONS[row_start:row_start + 2]):
            if col.button(suggestion, use_container_width=True):
                st.session_state.pending = suggestion

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed = st.chat_input("Ask about UGA resources...")
question = typed or st.session_state.pending
st.session_state.pending = None

# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------
if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    crisis = rx.is_crisis(question)

    # A short follow-up ("what about hours?") carries no retrievable terms, so
    # search against it combined with the previous question.
    retrieval_query = rx.resolve_query(question, history)
    matches = index.search(retrieval_query)

    if crisis:
        # Put crisis resources at the top of what gets cited, regardless of
        # how the rest of the query scored.
        crisis_rows = index.crisis_resources()
        others = matches[~matches["resource_name"].isin(crisis_rows["resource_name"])]
        matches = crisis_rows.assign(_score=99.0).head(4)
        if not others.empty:
            import pandas as pd

            matches = pd.concat([matches, others.head(2)])

    with st.chat_message("assistant"):
        if crisis:
            st.error(rx.CRISIS_NOTICE)

        if client is None:
            answer = rx.retrieval_answer(matches)
            st.markdown(answer)
        else:
            try:
                answer = st.write_stream(
                    stream_answer(client, model, question, matches, history, crisis)
                )
            except Exception as exc:
                answer = (
                    f"{rx.retrieval_answer(matches)}\n\n"
                    f"*(Couldn't reach OpenAI, so this is the raw search result. {exc})*"
                )
                st.markdown(answer)

        if not matches.empty:
            with st.expander(f"Resources referenced ({len(matches)})"):
                st.dataframe(
                    matches[["category", "resource_name", "description"]],
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption("Links open a search for the department's official page.")
                for name in matches["resource_name"]:
                    st.markdown(f"- [{name}]({rx.official_link(name)})")

    stored = rx.CRISIS_NOTICE + "\n\n---\n\n" + answer if crisis else answer
    st.session_state.messages.append({"role": "assistant", "content": stored})
