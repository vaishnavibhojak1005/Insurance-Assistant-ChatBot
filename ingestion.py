# ingestion.py
import os
import fitz  # PyMuPDF
from docx import Document
import chromadb
from sentence_transformers import SentenceTransformer # New: For local embeddings
from dotenv import load_dotenv
import re
import tiktoken

# Load environment variables
load_dotenv()

# Configuration
DOCS_DIR = "docs"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "insurance_policy_chunks"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
MAX_CHUNK_SIZE_TOKENS = 400
OVERLAP_TOKENS = 50

# --- Initialize Embedding Model ---
print(f"Loading SentenceTransformer model: {EMBEDDING_MODEL_NAME} for ingestion...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("SentenceTransformer model loaded for ingestion.")
except Exception as e:
    print(f"ERROR: Failed to load SentenceTransformer model: {e}")
    exit(1) # Exit if model fails to load

# --- Tokenizer ---
try:
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
except KeyError:
    tokenizer = tiktoken.get_encoding("cl100k_base")


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        document = fitz.open(pdf_path)
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text() + "\n"
        document.close()
        print(f"DEBUG: Extracted text length from {pdf_path}: {len(text)} characters.")
        if not text.strip():
            print(f"WARNING: No readable text extracted from {pdf_path}. Is it an image-only PDF?")
    except Exception as e:
        print(f"ERROR: Error extracting text from {pdf_path}: {e}")
    return text

def extract_text_from_docx(docx_path):
    text = ""
    try:
        document = Document(docx_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        print(f"DEBUG: Extracted text length from {docx_path}: {len(text)} characters.")
        if not text.strip():
            print(f"WARNING: No readable text extracted from {docx_path}.")
    except Exception as e:
        print(f"ERROR: Error extracting text from {docx_path}: {e}")
    return text

def clean_and_chunk_text(text, source_filename, max_chunk_tokens=MAX_CHUNK_SIZE_TOKENS, overlap_tokens=OVERLAP_TOKENS):
    if not text.strip():
        print(f"DEBUG: Skipping chunking for {source_filename} as text is empty.")
        return []

    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_chunk_tokens - overlap_tokens):
        chunk_tokens = tokens[i : i + max_chunk_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        if len(chunk_text.strip()) > 20:
            chunks.append({"text": chunk_text.strip(), "metadata": {"source": source_filename}})
    
    print(f"DEBUG: Generated {len(chunks)} chunks for {source_filename}.")
    if not chunks:
        print(f"WARNING: No valid chunks generated for {source_filename}. Check text content and chunking parameters.")
    return chunks

def get_embedding(text):
    try:
        embedding = embedding_model.encode([text])[0].tolist()
        # print(f"DEBUG: Generated embedding for text (first 5 chars): '{text[:5]}'...") # Too verbose for many chunks
        return embedding
    except Exception as e:
        print(f"ERROR: Failed to generate embedding for text (first 50 chars): '{text[:50]}...'. Error: {e}")
        return None


def main():
    if not os.path.exists(DOCS_DIR):
        print(f"ERROR: '{DOCS_DIR}' directory not found. Please create it and place your policy documents inside.")
        return

    all_chunks = []
    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".pdf"):
            print(f"[+] Processing PDF: {filename}")
            text = extract_text_from_pdf(filepath)
            all_chunks.extend(clean_and_chunk_text(text, filename))
        elif filename.endswith(".docx"):
            print(f"[+] Processing DOCX: {filename}")
            text = extract_text_from_docx(filepath)
            all_chunks.extend(clean_and_chunk_text(text, filename))
        else:
            print(f"[-] Skipping unsupported file: {filename}")

    if not all_chunks:
        print("ERROR: No processable documents found or no text extracted/chunked. Exiting.")
        return

    print(f"DEBUG: Total extracted and chunked {len(all_chunks)} text segments before embedding.")

    # Generate embeddings for all chunks
    print("DEBUG: Generating embeddings...")
    embeddings = []
    chunks_to_add = [] # Only add chunks that successfully get embeddings
    for i, chunk in enumerate(all_chunks):
        embedding = get_embedding(chunk['text'])
        if embedding is not None:
            embeddings.append(embedding)
            chunks_to_add.append(chunk)
        if (i + 1) % 100 == 0:
            print(f"DEBUG: Processed embeddings for {i+1}/{len(all_chunks)} chunks.")
    
    if not chunks_to_add:
        print("ERROR: No embeddings were successfully generated for any chunks. Exiting.")
        return

    print(f"DEBUG: Successfully generated embeddings for {len(chunks_to_add)} chunks (after filtering failures).")

    # Initialize ChromaDB client and collection
    print("DEBUG: Initializing ChromaDB client...")
    try:
        client_db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        print(f"DEBUG: ChromaDB client initialized for path: {CHROMA_DB_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to initialize ChromaDB client: {e}")
        exit(1)

    # Attempt to delete and recreate the collection for a fresh start
    try:
        # Check if the collection exists before attempting to delete
        existing_collections = client_db.list_collections()
        if any(c.name == COLLECTION_NAME for c in existing_collections):
            client_db.delete_collection(name=COLLECTION_NAME)
            print(f"DEBUG: Deleted existing collection: {COLLECTION_NAME}")
        else:
            print(f"DEBUG: Collection {COLLECTION_NAME} did not exist, creating new one.")
    except Exception as e:
        print(f"WARNING: Could not clear collection (maybe it didn't exist or an error occurred): {e}. Proceeding to get/create.")
    
    try:
        collection = client_db.get_or_create_collection(name=COLLECTION_NAME)
        print(f"DEBUG: ChromaDB collection '{COLLECTION_NAME}' accessed/created. Current count: {collection.count()}")
    except Exception as e:
        print(f"ERROR: Failed to access/create ChromaDB collection: {e}")
        exit(1)


    # Add documents to ChromaDB
    print("DEBUG: Preparing chunks for addition to ChromaDB...")
    documents = [chunk['text'] for chunk in chunks_to_add]
    metadatas = [chunk['metadata'] for chunk in chunks_to_add]
    ids = [f"doc_{i}" for i in range(len(chunks_to_add))]

    if not documents:
        print("WARNING: No documents prepared for addition to ChromaDB. This might indicate an earlier chunking/embedding issue.")
        print("[✓] Ingestion complete (but no data added).")
        return

    # ChromaDB can add in batches
    batch_size = 1000 # Increased batch size for potentially faster ingests
    total_added = 0
    for i in range(0, len(documents), batch_size):
        batch_documents = documents[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        try:
            collection.add(
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            total_added += len(batch_documents)
            print(f"DEBUG: Added batch {i // batch_size + 1}. Total added: {total_added} / {len(documents)}")
        except Exception as e:
            print(f"ERROR: Failed to add batch {i // batch_size + 1} to ChromaDB: {e}")
            break # Stop if adding fails

    final_count = collection.count()
    print(f"DEBUG: Final count in ChromaDB collection '{COLLECTION_NAME}': {final_count} chunks.")
    if final_count == 0:
        print("WARNING: ChromaDB collection is still empty after attempted ingestion.")

    print("[✓] Ingestion process finished.")

if __name__ == "__main__":
    main()