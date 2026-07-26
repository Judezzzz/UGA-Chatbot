"""Resource index, retrieval, and safety triage.

Deliberately free of Streamlit imports so everything here can be tested and
evaluated from a plain Python process — see test_resources.py and evaluate.py.

Retrieval is BM25 over a small hand-curated index, with query expansion for
the vocabulary mismatch problem (students say "therapist", the index says
"counseling") and a fuzzy fallback for typos.
"""

import difflib
import math
import re
from collections import Counter

import pandas as pd

DATA_FILE = "uga_resources.csv"
TOP_K = 6

# BM25 parameters. k1 controls how quickly repeated terms stop adding value;
# b controls how much longer documents are penalised. Standard defaults, and
# they behave well on documents this short.
BM25_K1 = 1.5
BM25_B = 0.75

# Fields are weighted by repetition when the searchable text is built, so a
# hit on the resource name counts for more than one in the description.
FIELD_WEIGHTS = {"resource_name": 4, "keywords": 3, "category": 2, "description": 1}

STOPWORDS = {
    "the", "and", "for", "are", "with", "who", "how", "what", "where", "when",
    "why", "can", "does", "did", "was", "were", "you", "your", "yours", "get",
    "got", "has", "have", "had", "any", "all", "there", "their", "they", "them",
    "this", "that", "these", "those", "from", "about", "into", "than", "then",
    "some", "please", "should", "would", "could", "make", "give", "tell",
    "know", "here", "its",
}

# Words a student is likely to use, mapped onto the vocabulary the index uses.
# This is the single biggest quality win for a small index: without it,
# "therapist" and "counseling" are unrelated strings.
SYNONYMS = {
    "therapist": ["counseling", "therapy", "mental"],
    "therapy": ["counseling", "mental"],
    "psychologist": ["counseling", "psychiatry", "mental"],
    "psychiatrist": ["psychiatry", "counseling", "mental"],
    "depressed": ["depression", "mental", "counseling"],
    "depression": ["mental", "counseling"],
    "anxious": ["anxiety", "mental", "counseling"],
    "anxiety": ["mental", "counseling"],
    "suicidal": ["suicide", "crisis", "emergency"],
    "suicide": ["crisis", "emergency"],
    "stressed": ["stress", "wellness", "counseling"],
    "burnout": ["stress", "wellness", "counseling"],
    "lonely": ["counseling", "community", "involvement"],
    "sick": ["health", "medical", "clinic"],
    "doctor": ["health", "medical", "clinic"],
    "medicine": ["pharmacy", "health", "medical"],
    "prescription": ["pharmacy", "health"],
    "hungry": ["food", "pantry", "groceries"],
    "starving": ["food", "pantry", "groceries"],
    "groceries": ["food", "pantry"],
    "broke": ["financial", "money", "aid"],
    "money": ["financial", "aid", "tuition"],
    "loan": ["financial", "aid", "fafsa"],
    "scholarship": ["financial", "aid", "funding"],
    "tuition": ["bursar", "financial", "aid", "bill"],
    "bill": ["bursar", "tuition", "payment"],
    "job": ["career", "employment", "internship"],
    "internship": ["career", "handshake", "employment"],
    "resume": ["career", "cover", "interview"],
    "interview": ["career", "coaching"],
    "hiring": ["career", "employers", "recruiting"],
    "tutor": ["tutoring", "academic", "support"],
    "homework": ["tutoring", "academic", "support"],
    "grades": ["academic", "advising", "athena"],
    "failing": ["academic", "tutoring", "advising"],
    "advisor": ["advising", "academic"],
    "schedule": ["registration", "advising", "athena"],
    "class": ["registration", "athena", "advising"],
    "register": ["registration", "athena"],
    "transcript": ["registrar", "records"],
    "dorm": ["housing", "residence"],
    "roommate": ["housing", "residence"],
    "apartment": ["housing", "legal", "lease"],
    "lease": ["legal", "housing"],
    "landlord": ["legal", "housing"],
    "eat": ["dining", "food", "meal"],
    "food": ["dining", "meal", "pantry"],
    "gym": ["recreation", "fitness", "ramsey"],
    "workout": ["recreation", "fitness", "ramsey"],
    "exercise": ["recreation", "fitness"],
    "bus": ["transit", "transportation"],
    "parking": ["permit", "transportation"],
    "car": ["parking", "permit"],
    "wifi": ["wireless", "network", "eits"],
    "internet": ["wireless", "network", "eits"],
    "password": ["myid", "eits", "account"],
    "computer": ["software", "eits", "technology"],
    "printing": ["bulldog", "bucks", "technology"],
    "club": ["organizations", "involvement", "activities"],
    "volunteer": ["service", "leadership", "community"],
    "research": ["curo", "undergraduate", "faculty"],
    "abroad": ["global", "international", "exchange"],
    "visa": ["international", "immigration"],
    "disability": ["accommodation", "accessibility", "drc"],
    "adhd": ["accommodation", "accessibility", "drc"],
    "accommodation": ["accessibility", "drc", "disability"],
    "harassment": ["title", "discrimination", "equity"],
    "assault": ["title", "crisis", "counseling", "police"],
    "discrimination": ["title", "equity", "harassment"],
    "cheating": ["conduct", "honesty", "academic"],
    "plagiarism": ["conduct", "honesty", "academic"],
    "drinking": ["alcohol", "fontaine", "substance"],
    "drugs": ["substance", "fontaine", "recovery"],
    "veteran": ["military", "benefits"],
    "lgbtq": ["lgbt", "gender", "sexuality"],
    "safety": ["police", "emergency", "safe"],
    "library": ["libraries", "study", "books"],
    "book": ["libraries", "books"],
    "essay": ["writing", "paper"],
    "paper": ["writing", "essay"],
    "math": ["mathematics", "calculus"],
}

# Phrases suggesting a student may be in danger. Matched against the raw
# question before tokenising, so multi-word signals survive.
CRISIS_PATTERNS = [
    r"\bkill (?:myself|my self)\b",
    r"\bkilling myself\b",
    r"\bsuicid",
    r"\bend (?:my|it) (?:life|all)\b",
    r"\bwant to die\b",
    r"\bdon'?t want to (?:live|be here)\b",
    r"\bhurt(?:ing)? myself\b",
    r"\bharm(?:ing)? myself\b",
    r"\bself[- ]harm",
    r"\boverdos",
    r"\braped?\b",
    r"\bsexual(?:ly)? assault",
    r"\bassaulted\b",
    r"\bin danger\b",
    r"\bbeing (?:stalked|followed|threatened)\b",
]

CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)


def is_crisis(question):
    """True when a question suggests immediate risk to safety.

    Intentionally errs toward false positives: showing crisis resources to
    someone who didn't need them is a mild annoyance, missing someone who
    did is not.
    """
    return bool(CRISIS_RE.search(question or ""))


def stem(word):
    """Very light suffix stripping — enough to join plural and gerund forms.

    Not a real stemmer and doesn't need to be: it only has to make
    "tutoring"/"tutor" and "loans"/"loan" collide in a 70-row index.
    """
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            if suffix == "ies":
                return word[: -len(suffix)] + "y"
            return word[: -len(suffix)]
    return word


def tokenise(text, apply_stem=True):
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    out = []
    for word in words:
        word = word.strip("'")
        if len(word) < 3 or word in STOPWORDS:
            continue
        out.append(stem(word) if apply_stem else word)
    return out


# Synonym lookup keyed by stem, built once so expansion is a dict hit.
_SYNONYMS_BY_STEM = {}
for _word, _expansions in SYNONYMS.items():
    _SYNONYMS_BY_STEM.setdefault(stem(_word), []).extend(stem(e) for e in _expansions)


# Synonyms count for less than what the student actually typed, so an
# expansion can promote the right resource without drowning the literal query.
SYNONYM_WEIGHT = 0.55


def expand(terms):
    """Weight the student's own terms, then add synonyms at a lower weight.

    Returns {term: weight}. Expansion happens *before* unknown terms are
    resolved against the vocabulary — do it the other way round and a word
    like "depressed" is discarded for not appearing in the index before its
    synonym ("counseling") ever gets a chance to fire.
    """
    weights = {}
    for term in terms:
        weights[term] = max(weights.get(term, 0), 1.0)
    for term in terms:
        for synonym in _SYNONYMS_BY_STEM.get(term, []):
            weights[synonym] = max(weights.get(synonym, 0), SYNONYM_WEIGHT)
    return weights


def searchable_text(row):
    parts = []
    for field, weight in FIELD_WEIGHTS.items():
        parts.extend([str(row.get(field, ""))] * weight)
    return " ".join(parts)


class ResourceIndex:
    """A small BM25 index over the resource table."""

    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.docs = [tokenise(searchable_text(row)) for _, row in self.frame.iterrows()]
        self.counts = [Counter(doc) for doc in self.docs]
        self.lengths = [len(doc) for doc in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0

        n_docs = len(self.docs) or 1
        doc_freq = Counter()
        for doc in self.docs:
            doc_freq.update(set(doc))
        # Standard BM25 idf, floored so very common terms can't score negative.
        self.idf = {
            term: max(0.05, math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)))
            for term, freq in doc_freq.items()
        }
        self.vocabulary = set(self.idf)

    def _resolve(self, weights):
        """Map terms the index has never seen onto ones it has.

        Two passes, cheapest first. A shared prefix catches the morphology the
        light stemmer misses ("depress" against "depression", "harass" against
        "harassment"); edit distance then catches ordinary typos.
        """
        resolved = {}
        for term, weight in weights.items():
            if term in self.vocabulary:
                resolved[term] = max(resolved.get(term, 0), weight)
                continue

            if len(term) >= 4:
                prefixed = [
                    v for v in self.vocabulary
                    if v.startswith(term) or (len(v) >= 4 and term.startswith(v))
                ]
                if prefixed:
                    best = min(prefixed, key=len)
                    resolved[best] = max(resolved.get(best, 0), weight * 0.9)
                    continue

            close = difflib.get_close_matches(term, self.vocabulary, n=1, cutoff=0.80)
            if close:
                resolved[close[0]] = max(resolved.get(close[0], 0), weight * 0.8)
        return resolved

    def score(self, weights):
        scores = []
        for i, counts in enumerate(self.counts):
            length = self.lengths[i] or 1
            total = 0.0
            for term, weight in weights.items():
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + BM25_K1 * (
                    1 - BM25_B + BM25_B * length / (self.avg_length or 1)
                )
                total += weight * idf * (tf * (BM25_K1 + 1)) / denom
            scores.append(total)
        return scores

    def search(self, question, limit=TOP_K, min_score=0.6):
        terms = tokenise(question)
        if not terms:
            return self.frame.iloc[0:0].assign(_score=pd.Series(dtype=float))

        weights = self._resolve(expand(terms))
        if not weights:
            return self.frame.iloc[0:0].assign(_score=pd.Series(dtype=float))

        ranked = self.frame.assign(_score=self.score(weights))
        hits = ranked[ranked["_score"] >= min_score].sort_values("_score", ascending=False)
        return hits.head(limit)

    def crisis_resources(self):
        return self.frame[self.frame["category"] == "Crisis"]

    @property
    def categories(self):
        return sorted(self.frame["category"].unique())

    def __len__(self):
        return len(self.frame)


def load_resources(path=DATA_FILE):
    return pd.read_csv(path).fillna("")


def build_index(path=DATA_FILE):
    return ResourceIndex(load_resources(path))


def official_link(resource_name):
    """A scoped search link rather than a hardcoded URL.

    Individual office URLs change and go stale; a search always resolves to
    something current and can never 404 the way a wrong hardcoded link would.
    """
    query = re.sub(r"\s+", "+", f"UGA {resource_name}".strip())
    return f"https://www.google.com/search?q={query}"


def format_matches(matches, links=True):
    lines = []
    for _, row in matches.iterrows():
        name = row["resource_name"]
        label = f"[{name}]({official_link(name)})" if links else name
        lines.append(f"- **{label}** ({row['category']}) — {row['description']}")
    return "\n".join(lines)


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


CRISIS_NOTICE = (
    "**If you are in immediate danger, call 911.**\n\n"
    "Free, confidential support is available 24/7:\n\n"
    "- **Call or text 988** — Suicide and Crisis Lifeline\n"
    "- **Text HOME to 741741** — Crisis Text Line\n\n"
    "UGA's Counseling and Psychiatric Services (CAPS) offers urgent same-day "
    "appointments for enrolled students during operating hours."
)

SYSTEM_PROMPT = (
    "You are a helpful assistant for University of Georgia students. "
    "Answer using ONLY the resources listed in the user message. "
    "Never invent an office, program, phone number, or web address. "
    "If the listed resources don't cover the question, say so plainly and "
    "point to the closest option. Be concise — a few sentences — and name "
    "the specific resources you're pointing to. Warm and direct, not chirpy."
)

CRISIS_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + " This student may be in distress or danger. Lead with care, keep it "
    "short and calm, and make sure they know they can call or text 988 at any "
    "time, or 911 if they are in immediate danger. Do not lecture, diagnose, "
    "or ask them to justify how they feel."
)


def build_prompt(question, matches, history=None):
    context = (
        format_matches(matches, links=False) if not matches.empty else "(no matches found)"
    )
    parts = []
    if history:
        recent = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history[-4:])
        parts.append(f"Earlier in this conversation:\n{recent}\n")
    parts.append(f"Available UGA resources:\n{context}\n")
    parts.append(f"Student question: {question}")
    return "\n".join(parts)


FOLLOWUP_STARTERS = (
    "what about", "how about", "and ", "what are", "where is", "when is",
    "who do", "is it", "are they", "do they", "hours", "cost", "price",
)


def needs_context(question):
    """Whether a question is too thin to retrieve on its own.

    "What are their hours?" carries no retrievable terms, so it should be
    resolved against whatever the student asked immediately before.
    """
    text = (question or "").strip().lower()
    if not text:
        return False
    # One content word is a fragment ("hours?"); two can already stand alone
    # ("find internship"), so only the former is treated as a follow-up.
    if len(tokenise(text)) <= 1:
        return True
    return text.startswith(FOLLOWUP_STARTERS)


def resolve_query(question, history):
    """Prepend the last student question when the current one can't stand alone."""
    if not history or not needs_context(question):
        return question
    for turn in reversed(history):
        if turn["role"] == "user":
            return f"{turn['content']} {question}"
    return question
