"""LLM provider selection.

Every provider here speaks the OpenAI wire format, so one client shape covers
all of them and the only differences are the base URL and the model name.
That keeps the app free to run: Groq and Google both offer real free tiers,
and OpenAI is supported for anyone who would rather pay for it.

No Streamlit import, so provider resolution can be tested directly. The
lookup function is injected rather than read from os.environ, because on
Streamlit Cloud keys arrive through st.secrets instead.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    name: str
    key_name: str
    model_key: str
    default_model: str
    label: str
    free: bool
    base_url: str = ""
    signup: str = ""

    def model(self, lookup):
        """Model id, overridable so a deprecated default can be fixed
        from the deploy settings without touching the code."""
        return lookup(self.model_key) or self.default_model


# Ordered by preference: free providers first, so a deployment with several
# keys configured defaults to the one that costs nothing.
PROVIDERS = [
    Provider(
        name="Groq",
        key_name="GROQ_API_KEY",
        model_key="GROQ_MODEL",
        default_model="llama-3.3-70b-versatile",
        label="Llama 3.3 70B via Groq",
        free=True,
        base_url="https://api.groq.com/openai/v1",
        signup="https://console.groq.com/keys",
    ),
    Provider(
        name="Gemini",
        key_name="GEMINI_API_KEY",
        model_key="GEMINI_MODEL",
        default_model="gemini-2.0-flash",
        label="Gemini 2.0 Flash",
        free=True,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        signup="https://aistudio.google.com/apikey",
    ),
    Provider(
        name="OpenAI",
        key_name="OPENAI_API_KEY",
        model_key="OPENAI_MODEL",
        default_model="gpt-4o-mini",
        label="GPT-4o mini",
        free=False,
        signup="https://platform.openai.com/api-keys",
    ),
]

FREE_PROVIDERS = [p for p in PROVIDERS if p.free]


def resolve(lookup):
    """Return (provider, api_key) for the first provider with a key set.

    Returns (None, None) when nothing is configured, which is a supported
    state — the app falls back to answering from the index directly.
    """
    for provider in PROVIDERS:
        key = lookup(provider.key_name)
        if key and str(key).strip():
            return provider, str(key).strip()
    return None, None


def make_client(provider, api_key):
    """Build an OpenAI-SDK client pointed at whichever provider was chosen."""
    from openai import OpenAI

    if provider.base_url:
        return OpenAI(api_key=api_key, base_url=provider.base_url)
    return OpenAI(api_key=api_key)
