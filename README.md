# Insurance-Assistant-ChatBot
🛡️ Insurance Document QA Assistant An AI-powered tool that parses insurance PDFs, embeds clauses using Universal Sentence Encoder, and retrieves relevant answers via semantic search with FAISS. Built with TensorFlow, Streamlit, and Python. Just upload a document, ask questions, and get instant, accurate answers.
Here’s a completed version of your `README.md`, following the style and structure you started with:


---

## 🚀 Features

* 📄 Parses uploaded insurance documents (PDF)
* 🧠 Uses Universal Sentence Encoder (USE) to embed clauses
* 🔍 Retrieves relevant clauses using vector similarity (FAISS)
* 💬 Answers user queries using semantic search
* 🖥️ Interactive web app powered by Streamlit
* 📈 Scalable architecture for multiple document support

---

## 📦 Tech Stack

* Python 3.10+
* TensorFlow 2.19
* TensorFlow Hub (Universal Sentence Encoder)
* FAISS (Facebook AI Similarity Search)
* Streamlit (for interactive UI)
* PyMuPDF / pdfplumber (for PDF parsing)

---

## 📁 Project Structure

```
insurance-doc-qa/
│
├── app/                          # Core application code
│   ├── embedding.py              # Embedding logic using USE
│   ├── retrieval.py              # FAISS vector search
│   ├── parser.py                 # PDF clause extraction
│   └── qa_engine.py              # Main question-answer logic
│
├── data/                         # Sample documents and embeddings
├── ui/                           # Streamlit UI components
├── utils/                        # Utility scripts
├── requirements.txt
├── streamlit_app.py              # Entry point for the Streamlit app
└── README.md
```

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/insurance-doc-qa.git
cd insurance-doc-qa
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## 🧪 Usage

1. Launch the web app.
2. Upload an insurance PDF document.
3. Ask a question (e.g., *"What is the deductible for dental coverage?"*).
4. The app retrieves the most relevant clause(s) and displays a concise answer.

---

## 📷 Screenshots (Optional)

You can include screenshots here using:

```markdown
![App Screenshot](screenshots/app_example.png)
```

---

## ❓ Example Queries

* *"What is the maximum out-of-pocket expense?"*
* *"Are pre-existing conditions covered?"*
* *"When does my coverage start?"*

---

## 📌 Future Improvements

* Integrate with OpenAI or LLaMA for advanced summarization
* Allow multi-document comparisons
* Add user authentication
* Support non-English documents

---

## 🛡️ Security & Privacy

* Local processing: documents and queries are handled securely on your machine
* No external API calls unless explicitly enabled



Let me know if you want help generating sample screenshots, setting up a Dockerfile, or converting this into a multi-page app with login/auth support.

