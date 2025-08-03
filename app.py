# app.py

import streamlit as st
from transformers import pipeline
from chatbot_logic import query_docs

# Load QA model (e.g., RoBERTa or any Hugging Face QA model)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

st.set_page_config(page_title="🛡️ Insurance Assistant", layout="centered")
st.title("🛡️ Insurance Policy Assistant")
st.markdown("Ask a question like: **'46-year-old male, knee surgery in Pune, 3-month-old insurance policy'**")

# Input
query = st.text_input("Enter your insurance-related question:")

if query:
    with st.spinner("Processing..."):
        result, chunks = query_docs(query)
        context = "\n".join([c["text"] for c in chunks])
        response = qa_pipeline(question=query, context=context)

        st.subheader("✅ Chatbot Decision")
        st.json(result)

        st.subheader("🧠 QA Model Answer")
        st.success(response.get("answer", "No answer found."))

        with st.expander("📄 Retrieved Policy Clauses"):
            for i, c in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}**: {c['text']}")


