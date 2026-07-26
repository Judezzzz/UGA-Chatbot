"""UGA Student Resource Chatbot.

Answers student questions about campus resources, grounded in the entries
in uga_resources.csv rather than the model's own recall — so it can't
invent an advising office that doesn't exist.
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

DATA_FILE = "uga_resources.csv"
MODEL = "gpt-3.5-turbo"

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
load_dotenv()  # local development reads .env


def get_api_key():
    """Streamlit Cloud injects st.secrets; locally we fall back to .env.

    Accessing st.secrets raises when no secrets file exists, which is the
    normal case on a laptop, so the lookup is guarded rather than assumed.
    """
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="UGA Student Resource Chatbot", page_icon="🐾")

st.title("UGA Student Resource Chatbot")
st.caption(
    "Ask about tutoring, advising, careers, or wellness. "
    "Answers are drawn from a curated list of UGA resources."
)


@st.cache_data
def load_resources():
    return pd.read_csv(DATA_FILE)


try:
    df = load_resources()
except FileNotFoundError:
    st.error(f"Could not find {DATA_FILE} next to app.py.")
    st.stop()

api_key = get_api_key()
if not api_key:
    st.warning(
        "No OpenAI API key found. Add `OPENAI_API_KEY` to your `.env` file "
        "locally, or to **Settings → Secrets** if this is running on "
        "Streamlit Community Cloud."
    )
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.subheader("Resources in this index")
    st.write(f"**{len(df)}** entries across **{df['category'].nunique()}** categories")
    st.dataframe(df, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def find_relevant(question, frame, limit=8):
    """Rank rows by how many question words they match.

    A plain keyword score rather than embeddings: the index is small enough
    that the extra dependency and latency wouldn't buy anything, and this
    stays easy to reason about. Falls back to the whole table when nothing
    matches, so the model still gets context instead of none.
    """
    words = {w.strip(".,?!").lower() for w in question.split() if len(w) > 3}
    if not words:
        return frame.head(limit)

    def score(row):
        haystack = " ".join(str(v).lower() for v in row.values)
        return sum(word in haystack for word in words)

    scored = frame.assign(_score=frame.apply(score, axis=1))
    hits = scored[scored["_score"] > 0].sort_values("_score", ascending=False)
    chosen = hits if not hits.empty else frame
    return chosen.drop(columns="_score", errors="ignore").head(limit)


def build_context(frame):
    return "\n".join(
        f"- [{row['category']}] {row['resource_name']}: {row['description']}"
        for _, row in frame.iterrows()
    )


SYSTEM_PROMPT = (
    "You are a helpful assistant for University of Georgia students. "
    "Answer using only the resources provided in the user message. "
    "If the resources do not cover the question, say so plainly and suggest "
    "the closest available option. Keep answers to a few sentences."
)

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about UGA resources...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    relevant = find_relevant(question, df)
    prompt = (
        f"Relevant UGA resources:\n{build_context(relevant)}\n\n"
        f"Student question: {question}"
    )

    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
            )
            answer = response.choices[0].message.content
        except Exception as exc:  # surface the failure instead of a blank reply
            answer = f"Sorry — I couldn't reach OpenAI just now. ({exc})"

        st.markdown(answer)
        with st.expander("Resources used"):
            st.dataframe(relevant, hide_index=True, use_container_width=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})
