import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv

# -----------------------------
# 1. Load API key from .env
# -----------------------------
load_dotenv()  # looks for .env file in current folder
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# -----------------------------
# 2. Streamlit Page Setup
# -----------------------------
st.title("UGA Student Resource Chatbot")
st.write("Ask a question about UGA resources (tutoring, advising, wellness, careers).")

# -----------------------------
# 3. Load CSV Resources
# -----------------------------
df = pd.read_csv("uga_resources.csv")

# -----------------------------
# 4. User Input
# -----------------------------
user_question = st.text_input("Your question:")

if user_question:

    # -----------------------------
    # 5. Build context from CSV
    # -----------------------------
    context = ""
    for _, row in df.iterrows():
        context += f"{row['category']}: {row['resource_name']} - {row['description']}\n"

    # -----------------------------
    # 6. Create GPT Prompt
    # -----------------------------
    prompt = f"""
You are a helpful assistant for UGA students.
Here are UGA resources:\n{context}\n
Answer the student's question: {user_question}
"""

    # -----------------------------
    # 7. Call OpenAI GPT
    # -----------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for UGA students."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )

        answer = response.choices[0].message.content
        st.write(answer)

    except Exception as e:
        st.error(f"Error contacting OpenAI: {e}")
