import streamlit as st
from PyPDF2 import PdfReader
import cohere
import numpy as np
import faiss

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main {
        background-color: #f9fafc;
        padding: 20px;
    }
    h1 {
        color: #2b6cb0;
        text-align: center;
        font-family: 'Arial Black', sans-serif;
    }
    .stButton>button {
        background-color: #2b6cb0;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #1e4e8c;
        color: white;
    }
    .stat-box {
        background: #edf2f7;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #cbd5e0;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Cohere client
COHERE_API_KEY = "j1fYvGNB4oiaPleuLLtLPp24tLgqgFhqI4BeAqEj"

@st.cache_resource
def get_cohere_client():
    return cohere.Client(COHERE_API_KEY)

# Load PDF with page numbers
def load_pdf(file):
    pdf = PdfReader(file)
    pages = []
    for i, page in enumerate(pdf.pages):
        page_text = page.extract_text()
        if page_text:
            pages.append({"page": i + 1, "text": page_text})
    return pages

# Load TXT (no pages, treat as single page)
def load_txt(file):
    text = file.read().decode("utf-8")
    return [{"page": 1, "text": text}]

# Chunk text while keeping page numbers
def chunk_text(pages, chunk_size=500, overlap=50):
    chunks = []
    for page in pages:
        words = page["text"].split()
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({"page": page["page"], "text": chunk_text})
            i += chunk_size - overlap
    return chunks

# Embed text with Cohere → return NumPy array
def embed_texts(co, texts):
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return np.array([], dtype=np.float32)
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    embeddings = np.array(response.embeddings, dtype=np.float32)
    return embeddings

# Build FAISS index
def build_faiss_index(embeddings):
    if embeddings.size == 0:
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Generate answer from context
def generate_answer(co, context_chunks, question,
                    model="command-r-plus",
                    temperature=0.3):
    context = "\n\n".join([c["text"] for c in context_chunks])
    prompt = (
        f"You are a helpful assistant. Use the provided context to answer accurately.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = co.chat(
        model=model,
        message=prompt,
        temperature=temperature
    )
    return response.text.strip()

# Main App
def main():
    st.title("📚 RAG Chatbot with Page Numbers")

    co = get_cohere_client()

    uploaded_file = st.file_uploader("📤 Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            pages = load_pdf(uploaded_file)
        else:
            pages = load_txt(uploaded_file)

        total_text = sum(len(p["text"]) for p in pages)
        st.markdown(f'<div class="stat-box">📄 Document Loaded: <b>{total_text} characters</b></div>', unsafe_allow_html=True)

        chunks = chunk_text(pages)
        st.markdown(f'<div class="stat-box">✂️ Split into <b>{len(chunks)}</b> chunks</div>', unsafe_allow_html=True)

        # Create embeddings for the entire document (optional)
        with st.spinner("⚡ Creating embeddings..."):
            embeddings = embed_texts(co, [c["text"] for c in chunks])

        if embeddings.size == 0:
            st.error("❌ No text found to embed. Please upload a valid file.")
            return

        st.success("✅ Embeddings created successfully")

        # Save to session
        st.session_state['chunks'] = chunks
        st.session_state['co'] = co

    # Query Section
    if 'chunks' in st.session_state:
        page_numbers = sorted(set(c["page"] for c in st.session_state['chunks']))
        selected_page = st.number_input(
            "📄 Enter page number to search",
            min_value=min(page_numbers),
            max_value=max(page_numbers),
            value=min(page_numbers)
        )

        query = st.text_input("🔍 Enter your query")
        
        if query:
            # Filter chunks for selected page
            chunks_on_page = [c for c in st.session_state['chunks'] if c["page"] == selected_page]

            if not chunks_on_page:
                st.error(f"❌ No content found on page {selected_page}.")
            else:
                # Embed only the chunks on that page
                embeddings_page = embed_texts(st.session_state['co'], [c["text"] for c in chunks_on_page])
                index_page = build_faiss_index(embeddings_page)

                # Embed query
                q_embedding = embed_texts(st.session_state['co'], [query])
                if q_embedding.size == 0:
                    st.error("❌ Query embedding failed.")
                else:
                    # Search only within this page
                    D, I = index_page.search(q_embedding, k=min(3, len(chunks_on_page)))

                    # Collect relevant chunks
                    relevant_chunks = [chunks_on_page[idx] for idx in I[0] if 0 <= idx < len(chunks_on_page)]

                    # Generate answer
                    answer = generate_answer(st.session_state['co'], relevant_chunks, query)

                    # Show answer + page
                    st.success("✅ Answer generated:")
                    st.write(answer)
                    st.info(f"📄 Page: {selected_page}")

if __name__ == "__main__":
    main()
