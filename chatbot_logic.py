# chatbot_logic.py

import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import json
import re

# Load environment variables (optional if not using API keys)
load_dotenv()

# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "insurance_policy_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local sentence-transformers model
TOP_K_CHUNKS = 5

# Load local embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def get_embedding(text):
    return embedding_model.encode(text)

def get_db_collection():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return None

def parse_query(query):
    # Very basic entity extraction (regex-based). Improve as needed.
    parsed = {
        "age": None,
        "gender": None,
        "procedure": None,
        "location": None,
        "policy_duration": None,
        "original_query": query
    }

    age_match = re.search(r"(\d{1,3})[- ]?year[- ]?old", query.lower())
    if age_match:
        parsed["age"] = int(age_match.group(1))

    if "male" in query.lower():
        parsed["gender"] = "male"
    elif "female" in query.lower():
        parsed["gender"] = "female"

    loc_match = re.search(r"in ([a-zA-Z\s]+)", query)
    if loc_match:
        parsed["location"] = loc_match.group(1).strip()

    proc_match = re.search(r", (.+?) in", query)
    if proc_match:
        parsed["procedure"] = proc_match.group(1).strip()

    dur_match = re.search(r"(\d+[- ]?(month|year)[s]?)", query.lower())
    if dur_match:
        parsed["policy_duration"] = dur_match.group(1).replace("-", " ")

    return parsed

def retrieve_chunks(query, collection, k=TOP_K_CHUNKS):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]

def reason_decision(query, parsed, context_chunks):
    context_text = "\n".join([c["text"] for c in context_chunks])

    decision = "Undetermined"
    amount = "N/A"
    justification = "Insufficient context to confidently determine eligibility."
    matched_clauses = []

    for chunk in context_chunks:
        text = chunk["text"].lower()

        if parsed["procedure"] and parsed["procedure"].lower() in text:
            matched_clauses.append(chunk["text"])
            if "not covered" in text or "excluded" in text:
                decision = "Rejected"
                justification = f"Procedure appears to be excluded based on clause: '{chunk['text'][:100]}...'"
                break
            elif "covered" in text or "eligible" in text:
                decision = "Approved"
                justification = f"Procedure seems eligible under clause: '{chunk['text'][:100]}...'"
                break

    return {
        "Decision": decision,
        "Amount": amount,
        "Justification": justification,
        "ApplicableClauses": matched_clauses
    }

def query_docs(query):
    collection = get_db_collection()
    parsed = parse_query(query)
    chunks = retrieve_chunks(query, collection)
    result = reason_decision(query, parsed, chunks)
    return result, chunks
