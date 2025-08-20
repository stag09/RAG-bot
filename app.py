import streamlit as st
import pdfplumber
import cohere
import numpy as np
import faiss
import time
import random
from cohere.errors import TooManyRequestsError

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main { background-color: #f9fafc; padding: 20px; }
    h1 { color: #2b6cb0; text-align: center; font-family: 'Arial Black', sans-serif; }
    .stButton>button {
        background-color: #2b6cb0; color: white; font-size: 16px;
        border-radius: 8px; padding: 10px 20px;
    }
    .stButton>button:hover { background-color: #1e4e8c; color: white; }
    .stat-box {
        background: #edf2f7; padding: 10px; border-radius: 8px;
        border: 1px solid #cbd5e0; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Cohere client
COHERE_API_KEY = "j1fYvGNB4oiaPleuLLtLPp24tLgqgFhqI4BeAqEj"
@st.cache_resource
def get_cohere_client():
    return cohere.Client(COHERE_API_KEY)

# Load PDF with pdfplumber (better than PyPDF2)
def load_pdf(file):
    pages = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append({"page": i + 1, "text": page_text})
    return pages

# Load TXT
def load_txt(file):
    text = file.read().decode("utf-8")
    return [{"page": 1, "text": text}] if text.strip() else []

# Chunk text
def chunk_text(pages, chunk_size=500, overlap=50):
    chunks = []
    for page in pages:
        words = page["text"].split()
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:  # skip empty
                chunks.append({"page": page["page"], "text": chunk_text})
            i += chunk_size - overlap
    return chunks

# Embed text with Cohere (batch=500)
def embed_texts(co, texts, batch_size=500, max_retries=5):
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return np.array([], dtype=np.float32)

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for attempt in range(max_retries):
            try:
                response = co.embed(
                    texts=batch,
                    model="embed-english-light-v3.0",
                    input_type="search_document"
                )
                embeddings.extend(response.embeddings)
                break
            except TooManyRequestsError:
                time.sleep((2 ** attempt) + random.random())
    return np.array(embeddings, dtype=np.float32)

# Build FAISS
def build_faiss_index(embeddings):
    if embeddings.size == 0:
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Generate answer
def generate_answer(co, context_chunks, question,
                    model="command-r-plus",
                    temperature=0.3):
    context = "\n\n".join([c["text"] for c in context_chunks])
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = co.chat(model=model, message=prompt, temperature=temperature)
    return response.text.strip()

# Main App
def main():
    st.title("📚 RAG Chatbot ")

    co = get_cohere_client()

    uploaded_file = st.file_uploader("📤 Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            pages = load_pdf(uploaded_file)
        else:
            pages = load_txt(uploaded_file)

        if not pages:
            st.error("❌ No text found in document.")
            return

        total_text = sum(len(p["text"]) for p in pages)
        st.markdown(f'<div class="stat-box">📄 Document Loaded: <b>{total_text} characters</b></div>', unsafe_allow_html=True)

        chunks = chunk_text(pages)
        st.markdown(f'<div class="stat-box">✂️ Split into <b>{len(chunks)}</b> chunks</div>', unsafe_allow_html=True)

        with st.spinner("⚡ Creating embeddings..."):
            embeddings = embed_texts(co, [c["text"] for c in chunks])

        if embeddings.size == 0:
            st.error("❌ Embedding failed. Try another file.")
            return

        st.success("✅ Embeddings created successfully")

        index = build_faiss_index(embeddings)
        if index is None:
            st.error("❌ Failed to build FAISS index.")
            return

        st.session_state['chunks'] = chunks
        st.session_state['index'] = index
        st.session_state['embeddings'] = embeddings
        st.session_state['co'] = co

    query = st.text_input("🔍 Enter your query")
    if query and 'index' in st.session_state:
        with st.spinner("🔎 Searching..."):
            q_embedding = embed_texts(st.session_state['co'], [query], batch_size=1)
            if q_embedding.size == 0:
                st.error("❌ Query embedding failed.")
            else:
                D, I = st.session_state['index'].search(q_embedding, k=3)
                relevant_chunks = [st.session_state['chunks'][idx] for idx in I[0] if 0 <= idx < len(st.session_state['chunks'])]
                answer = generate_answer(st.session_state['co'], relevant_chunks, query)
                pages_used = [c["page"] for c in relevant_chunks]
                st.success("✅ Answer generated:")
                st.write(answer)
                st.info(f"📄 Found on page(s): {sorted(set(pages_used))}")

if __name__ == "__main__":
    main()
