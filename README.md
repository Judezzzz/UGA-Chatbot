# UGA Student Resource Chatbot

A Streamlit chatbot that helps University of Georgia students find campus
resources — advising, tutoring, health, money, housing, safety, and more.

Answers are **grounded in `uga_resources.csv`** (66 resources across 15
categories). The app retrieves the entries most relevant to a question and
instructs the model to answer only from those, so it can't invent an office
that doesn't exist. Every reply shows the rows it used.

**It works without an API key.** With no key configured it runs in *search
mode* and answers directly from the index, so a deployed demo is useful to
anyone who opens it rather than a dead page behind someone else's billing.

## Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — chat, sidebar browser, suggested questions |
| `resources.py` | Index loading and retrieval, kept UI-free so it's testable |
| `test_resources.py` | Checks for the index and retrieval quality |
| `uga_resources.csv` | The index (`category`, `resource_name`, `description`, `keywords`) |
| `requirements.txt` | Pinned dependencies — required for Streamlit Cloud to build |

## How retrieval works

Each row is flattened into one searchable string, with the stronger fields
repeated so a hit on the resource name outweighs a hit in the description:

```
name x 4  +  keywords x 3  +  category x 2  +  description
```

A question is tokenised, stopwords are dropped, and rows are scored by how
many query terms they contain — whole-word matches counting for more than
substring matches. The top 6 rows become the model's context.

Deliberately not embeddings: the index is small enough that term matching
performs well, and this adds no dependency, no latency, and no cost. It is
also easy to debug — you can see exactly why a row matched.

When nothing matches, the app says so rather than showing unrelated rows.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

That's enough to use it in search mode. For model-written answers, add a key:

```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

## Tests

```bash
python test_resources.py
```

Covers index integrity (no blanks, no duplicates, 50+ entries, 10+
categories), tokenising, a dozen realistic questions mapped to the resource
that should surface, and edge cases like nonsense and stopword-only queries.

## Deploying to Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → `judezekra/UGA-Chatbot`, branch `main`, file `app.py`.
3. **Deploy.** It will run in search mode immediately.
4. Optional — for model-written answers, open **Settings → Secrets** and add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```

Keys belong in Streamlit's Secrets, never in the repo. This is a public
repository and committed keys get scraped within minutes.

## Adding or correcting resources

Append rows to `uga_resources.csv` — no code changes needed. The sidebar
count, category filter, and retrieval all read from the file.

The index is student-maintained and descriptions are intentionally general;
it points students toward the right office rather than acting as an official
directory. Verify specifics against each department's own page.
