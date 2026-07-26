# UGA Student Resource Chatbot

A Streamlit chatbot that helps University of Georgia students find campus
resources — advising, tutoring, health, money, housing, safety, and more.

**[Live demo →](https://uga-resources.streamlit.app/)**

Answers are grounded in a curated index of **70 resources across 16
categories**. The app retrieves the entries most relevant to a question and
instructs the model to answer only from those, so it cannot invent an office
that doesn't exist. Every reply cites the rows behind it.

---

## Three things that make it work

**1. Bring any model key, or none at all.** Groq, Google, and OpenAI all
speak the OpenAI wire format, so the app supports all three through one
client — only the base URL and model name differ. Free providers are
preferred when several keys are present, and with no key at all it answers
directly from the index in *search mode*, so a public demo stays useful
instead of being a dead page behind someone else's billing.

**2. Questions suggesting danger short-circuit to crisis resources.** Before
retrieval runs, the question is checked against phrases indicating risk to
safety. If matched, 988, Crisis Text Line, emergency services, and CAPS are
surfaced at the top regardless of how the rest of the query scored, and the
model is re-instructed to lead with care. This deliberately errs toward false
positives — showing crisis resources to someone who didn't need them is a
mild annoyance; missing someone who did is not.

**3. Retrieval is measured, not assumed.** `evaluate.py` scores the index
against 59 labelled questions written the way students actually type them,
vague phrasings and typos included.

```
Recall@1   89.8%
Recall@3   100.0%
Recall@5   100.0%
MRR        0.949
```

---

## How retrieval works

**BM25** over a searchable string per resource, with stronger fields repeated
so a name match outranks a description match:

```
name x 4  +  keywords x 3  +  category x 2  +  description
```

BM25 rather than raw term counting because it weights rare terms by inverse
document frequency — "pantry" should count for far more than "student" — and
normalises for document length so verbose entries don't win by being long.

Three layers then close the gap between a student's wording and the index's:

| Layer | Problem it solves | Example |
|---|---|---|
| Light stemming | Plural and gerund forms | `loans` to `loan`, `tutoring` to `tutor` |
| Synonym expansion | Vocabulary mismatch | `therapist` to `counseling`, `hungry` to `pantry` |
| Prefix + edit distance | Morphology and typos | `depress` to `depression`, `councelling` to `counseling` |

Synonyms are added at 0.55 weight so an expansion can promote the right
resource without drowning out what the student literally typed.

Order matters here: expansion happens **before** unknown terms are resolved.
Doing it the other way discards a word like "depressed" for not appearing in
the index before its synonym ever fires — that was a real bug the evaluation
caught, worth 15 points of Recall@1.

Deliberately not embeddings: at 70 documents, term matching with these layers
measures at 100% Recall@3 while adding no dependency, no latency, no cost,
and staying debuggable — you can see exactly why a row matched.

When nothing clears the score threshold, the app says so rather than
presenting unrelated rows as answers.

---

## Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — chat, streaming, crisis banner, sidebar browser |
| `providers.py` | Model provider config and selection, Streamlit-free |
| `resources.py` | Index, BM25 retrieval, synonyms, crisis detection, follow-up resolution. No Streamlit import, so it stays testable |
| `test_resources.py` | 45 behaviour checks — index integrity, retrieval, crisis triage, follow-ups |
| `evaluate.py` | Retrieval quality metrics over labelled questions |
| `uga_resources.csv` | The index (`category`, `resource_name`, `description`, `keywords`) |
| `.streamlit/config.toml` | Theme |

## Other behaviour

- **Follow-up questions.** "What about hours?" carries no retrievable terms,
  so a question with only one content word is searched together with the
  previous one.
- **Streaming replies** when a key is present, falling back to the search
  result if the API call fails rather than erroring out.
- **Links are searches, not hardcoded URLs.** Office URLs go stale; a scoped
  search always resolves to something current and can never 404.

## Model providers

Any one of these keys turns on written answers. Both free options issue a key
in under a minute and neither asks for a card.

| Provider | Key | Default model | Cost | Get a key |
|---|---|---|---|---|
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | Free tier | [console.groq.com/keys](https://console.groq.com/keys) |
| Google | `GEMINI_API_KEY` | `gemini-2.0-flash` | Free tier | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` | Paid | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

Free tiers are rate-limited rather than unlimited; the app falls back to
search results if a call fails, so hitting a limit degrades instead of
breaking.

Each provider's model is overridable — set `GROQ_MODEL`, `GEMINI_MODEL`, or
`OPENAI_MODEL` — so a deprecated default can be fixed from the deploy
settings without a code change.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py          # search mode, no key needed
python test_resources.py      # behaviour checks
python evaluate.py            # retrieval metrics
```

For written answers, add whichever key you have:

```bash
echo "GROQ_API_KEY=gsk-your-key-here" > .env
```

## Deploying

Streamlit Community Cloud → **New app** → `judezekra/UGA-Chatbot`, branch
`main`, file `app.py`. It runs in search mode immediately.

For written answers, open **Settings → Secrets** and add:

```toml
GROQ_API_KEY = "gsk-your-key-here"
```

Keys belong in Secrets, never in the repo — this one is public and gets
scraped.

## Adding or correcting resources

Append rows to `uga_resources.csv`; no code changes needed. Then run
`evaluate.py` to confirm nothing regressed, and add a case there for whatever
the new resource should answer.

The index is student-maintained and descriptions are intentionally general —
it points students toward the right office rather than acting as an official
directory. Verify specifics against each department's own page.
