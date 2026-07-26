"""Resource index and retrieval.

Kept separate from the Streamlit UI so the matching logic can be tested
without spinning up a server — see test_resources.py.
"""

import re

import pandas as pd

DATA_FILE = "uga_resources.csv"
TOP_K = 6

# Words that appear in almost any question and would otherwise dominate scoring.
STOPWORDS = {
    "the", "and", "for", "are", "with", "who", "how", "what", "where", "when",
    "why", "can", "does", "did", "was", "were", "you", "your", "yours", "get",
    "got", "has", "have", "had", "any", "all", "there", "their", "they", "them",
    "this", "that", "these", "those", "from", "about", "into", "than", "then",
    "some", "help", "need", "want", "find", "uga", "student", "students",
    "campus", "school", "college", "university", "please", "should", "would",
    "could", "make", "give", "tell", "know", "here",
}


def load_resources(path=DATA_FILE):
    frame = pd.read_csv(path).fillna("")
    # One prebuilt haystack per row. Stronger fields are repeated so a hit on
    # the resource name counts for more than a hit in the description.
    frame["_haystack"] = (
        (frame["resource_name"] + " ") * 4
        + (frame["keywords"] + " ") * 3
        + (frame["category"] + " ") * 2
        + frame["description"]
    ).str.lower()
    return frame


def tokenise(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def search(question, frame, limit=TOP_K):
    """Rank rows by how many query terms they contain.

    Plain term matching rather than embeddings: the index is small, this adds
    no dependency or latency, and it stays easy to explain and debug. Returns
    an empty frame when nothing matches, so callers can say so honestly
    instead of presenting unrelated rows as answers.
    """
    terms = tokenise(question)
    if not terms:
        return frame.iloc[0:0]

    def score(haystack):
        padded = f" {haystack} "
        return sum(
            3 if f" {term} " in padded else 1
            for term in terms
            if term in haystack
        )

    scored = frame.assign(_score=frame["_haystack"].map(score))
    hits = scored[scored["_score"] > 0].sort_values("_score", ascending=False)
    return hits.head(limit)


def format_matches(matches):
    return "\n".join(
        f"- **{row['resource_name']}** ({row['category']}) — {row['description']}"
        for _, row in matches.iterrows()
    )


def retrieval_answer(matches):
    """Deterministic reply used when no API key is configured."""
    if matches.empty:
        return (
            "I couldn't find anything matching that in the resource index. "
            "Try different wording, or browse the categories in the sidebar."
        )
    lead = (
        "Here's what I found in the resource index:"
        if len(matches) > 1
        else "Here's the closest match:"
    )
    return f"{lead}\n\n{format_matches(matches)}"


SYSTEM_PROMPT = (
    "You are a helpful assistant for University of Georgia students. "
    "Answer using ONLY the resources listed in the user message. "
    "Never invent an office, program, phone number, or web address. "
    "If the listed resources don't cover the question, say so plainly and "
    "point to the closest option. Be concise — a few sentences — and name "
    "the specific resources you're pointing to."
)


def build_prompt(question, matches):
    context = format_matches(matches) if not matches.empty else "(no matches found)"
    return f"Available UGA resources:\n{context}\n\nStudent question: {question}"
