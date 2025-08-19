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
    .chunk-box {
        background: #f7fafc;
        padding: 8px;
        border-left: 4px solid #2b6cb0;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Cohere client
COHERE_API_KEY = "j1fYvGNB4oiaPleuLLtLPp24tLgqgFhqI4BeAqEj"
@st.cache_resource
def get_cohere_client():
    return cohere.Client(COHERE_API_KEY)

# Load PDF
def load_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Load TXT
def load_txt(file):
    return file.read().decode("utf-8")

# Chunk text
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
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
    context = "\n\n".join(context_chunks)
    prompt = (
        f"You are a helpful assistant. Use the provided context to answer accurately.,i want proved page.no also give me \n\n"
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
    st.title("📚 RAG Chatbot")

    co = get_cohere_client()

    uploaded_file = st.file_uploader("📤 Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = load_pdf(uploaded_file)
        else:
            text = load_txt(uploaded_file)

        st.markdown(f'<div class="stat-box">📄 Document Loaded: <b>{len(text)} characters</b></div>', unsafe_allow_html=True)

        chunks = chunk_text(text)
        st.markdown(f'<div class="stat-box">✂️ Split into <b>{len(chunks)}</b> chunks</div>', unsafe_allow_html=True)

        # Create embeddings
        with st.spinner("⚡ Creating embeddings..."):
            embeddings = embed_texts(co, chunks)

        if embeddings.size == 0:
            st.error("❌ No text found to embed. Please upload a valid file.")
            return

        st.success("✅ Embeddings created successfully")

        # Build FAISS index
        index = build_faiss_index(embeddings)
        if index is None:
            st.error("❌ Failed to build FAISS index.")
            return

        # Save to session
        st.session_state['chunks'] = chunks
        st.session_state['index'] = index
        st.session_state['embeddings'] = embeddings
        st.session_state['co'] = co

    # Query
    query = st.text_input("🔍 Enter your query")
    if query and 'index' in st.session_state:
         with st.spinner("🔎 Searching..."):
            q_embedding = embed_texts(st.session_state['co'], [query])
            if q_embedding.size == 0:
                st.error("❌ Query embedding failed.")
            else:
                D, I = st.session_state['index'].search(q_embedding, k=3)

            # Collect relevant chunks (used internally only)
                relevant_chunks = []
                for idx in I[0]:
                    if 0 <= idx < len(st.session_state['chunks']):
                        relevant_chunks.append(st.session_state['chunks'][idx])

            # Generate answer
            answer = generate_answer(
                st.session_state['co'],
                relevant_chunks,
                query
            )

            # Show only the final answer
            st.success("✅ Answer generated:")
            st.write(answer)


if __name__ == "__main__":
    main()
