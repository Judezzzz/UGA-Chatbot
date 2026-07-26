"""UGA Student Resource Chatbot.

Answers questions about University of Georgia campus resources from a
curated index rather than the model's own recall, so it cannot invent an
office that doesn't exist. Every answer shows the entries it drew from.

Runs with or without an OpenAI key: without one it falls back to pure
retrieval, so a deployed demo still works for anyone who opens it.

Retrieval lives in resources.py so it can be tested without Streamlit.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from resources import (
    DATA_FILE,
    SYSTEM_PROMPT,
    build_prompt,
    load_resources,
    retrieval_answer,
    search,
)

MODEL = "gpt-3.5-turbo"

load_dotenv()


def get_api_key():
    """Streamlit Cloud injects st.secrets; local runs use .env.

    Reading st.secrets raises when no secrets file exists, which is the
    normal case on a laptop, so the lookup is guarded rather than assumed.
    """
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def llm_answer(client, question, matches):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, matches)},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content


st.set_page_config(page_title="UGA Student Resource Chatbot", page_icon="🐾")


@st.cache_data
def cached_resources():
    return load_resources()


try:
    df = cached_resources()
except FileNotFoundError:
    st.error(f"Could not find {DATA_FILE} next to app.py.")
    st.stop()

api_key = get_api_key()
client = None
if api_key:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
    except Exception as exc:
        st.warning(f"Could not start the OpenAI client, using search only. ({exc})")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("UGA Student Resource Chatbot")
st.caption(
    "Ask about advising, tutoring, health, money, housing, or campus life. "
    f"Answers are grounded in an index of {len(df)} UGA resources."
)

if client is None:
    st.info(
        "Running in **search mode** — no OpenAI key is configured, so replies "
        "list matching resources directly instead of being written by a model. "
        "Everything else works the same.",
        icon="🔍",
    )

with st.sidebar:
    st.subheader("Browse the index")
    st.caption(f"{len(df)} resources across {df['category'].nunique()} categories")
    chosen = st.selectbox("Category", ["All"] + sorted(df["category"].unique()))
    view = df if chosen == "All" else df[df["category"] == chosen]
    st.dataframe(
        view[["resource_name", "description"]],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "A student-maintained index, not an official UGA directory. "
        "Confirm details on the department's own page."
    )

SUGGESTIONS = [
    "Where can I get free tutoring?",
    "I'm struggling with my mental health",
    "How do I find an internship?",
    "I can't afford groceries this month",
]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

if not st.session_state.messages:
    st.write("**Try one of these:**")
    for col, suggestion in zip(st.columns(2), SUGGESTIONS[:2]):
        if col.button(suggestion, use_container_width=True):
            st.session_state.pending = suggestion
    for col, suggestion in zip(st.columns(2), SUGGESTIONS[2:]):
        if col.button(suggestion, use_container_width=True):
            st.session_state.pending = suggestion

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed = st.chat_input("Ask about UGA resources...")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    matches = search(question, df)

    with st.chat_message("assistant"):
        if client is None:
            answer = retrieval_answer(matches)
        else:
            try:
                answer = llm_answer(client, question, matches)
            except Exception as exc:
                # Fall back to the search result rather than failing outright.
                answer = (
                    f"{retrieval_answer(matches)}\n\n"
                    f"*(Couldn't reach OpenAI, so this is the raw search result. {exc})*"
                )

        st.markdown(answer)

        if not matches.empty:
            with st.expander(f"Resources used ({len(matches)})"):
                st.dataframe(
                    matches[["category", "resource_name", "description"]],
                    hide_index=True,
                    use_container_width=True,
                )

    st.session_state.messages.append({"role": "assistant", "content": answer})
