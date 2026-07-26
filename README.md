# UGA Student Resource Chatbot

A Streamlit chatbot that answers University of Georgia students' questions
about campus resources — tutoring, advising, careers, wellness.

Answers are **grounded in `uga_resources.csv`**: the app selects the entries
most relevant to each question and instructs the model to answer only from
those, so it can't invent an office that doesn't exist. Every reply shows the
rows it used.

## Running locally

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-your-key-here" > .env
streamlit run app.py
```

## Deploying to Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → pick `judezekra/UGA-Chatbot`, branch `main`, file `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
4. **Deploy.** You'll get a public `*.streamlit.app` URL.

The key must go in Streamlit's Secrets, never in the repo — this is a public
repository, and committed keys get scraped within minutes.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app: retrieval + chat interface |
| `uga_resources.csv` | The resource index (`category`, `resource_name`, `description`) |
| `requirements.txt` | Pinned dependencies — required for Streamlit Cloud to build |

## Adding resources

Append rows to `uga_resources.csv`. No code changes needed; the sidebar count
and the retrieval step both read from the file directly.
