import streamlit as st
from PyPDF2 import PdfReader
import cohere
import numpy as np
import faiss
from cohere.errors import TooManyRequestsError

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main { background-color: #f9fafc; padding: 20px; }
    h1 { color: #2b6cb0; text-align: center; font-family: 'Arial Black', sans-serif; }
    .stButton>button { background-color: #2b6cb0; color: white; font-size: 16px; border-radius: 8px; padding: 10px 20px; }
    .stButton>button:hover { background-color: #1e4e8c; color: white; }
    .stat-box { background: #edf2f7; padding: 10px; border-radius: 8px; border: 1px solid #cbd5e0; margin-bottom: 10px; }
    .source-box { background: #f7fafc; padding: 10px; border-left: 4px solid #2b6cb0; margin-bottom: 10px; font-size: 14px; }
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
    return [{"page": i + 1, "text": page.extract_text()} for i, page in enumerate(pdf.pages) if page.extract_text()]

# Load TXT (treat as single page)
def load_txt(file):
    text = file.read().decode("utf-8")
    return [{"page": 1, "text": text}]

# Chunk text while keeping page numbers
def chunk_text(pages, chunk_size=400, overlap=40):
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

# Embed text with Cohere → return NumPy array (fast mode, no waiting)
def embed_texts(co, texts, batch_size=32, max_retries=2):
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
                    model="embed-english-light-v3.0",  # ⚡ faster embeddings
                    input_type="search_document"
                )
                embeddings.extend(response.embeddings)
                break
            except TooManyRequestsError:
                if attempt == max_retries - 1:
                    st.error("⚠️ Too many requests, embedding failed.")
    return np.array(embeddings, dtype=np.float32)

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
                    temperature=0.2):
    context = "\n\n".join([c["text"] for c in context_chunks])
    prompt = (
        f"Answer based on context only.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = co.chat(
        model=model,
        message=prompt,
        temperature=temperature,
        max_tokens=300  # ⚡ shorter, faster
    )
    return response.text.strip()

# Main App
def main():
    st.title("📚 Fast PDF/TXT Q&A with Cohere")

    co = get_cohere_client()

    uploaded_file = st.file_uploader("📤 Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded_file and 'index' not in st.session_state:
        if uploaded_file.type == "application/pdf":
            pages = load_pdf(uploaded_file)
        else:
            pages = load_txt(uploaded_file)

        total_text = sum(len(p["text"]) for p in pages)
        st.markdown(f'<div class="stat-box">📄 Loaded: <b>{total_text} characters</b></div>', unsafe_allow_html=True)

        chunks = chunk_text(pages)
        st.markdown(f'<div class="stat-box">✂️ Split into <b>{len(chunks)}</b> chunks</div>', unsafe_allow_html=True)

        with st.spinner("⚡ Creating embeddings..."):
            embeddings = embed_texts(co, [c["text"] for c in chunks])

        if embeddings.size == 0:
            st.error("❌ No text found to embed. Upload a valid file.")
            return

        index = build_faiss_index(embeddings)
        if index is None:
            st.error("❌ Failed to build FAISS index.")
            return

        st.session_state.update({
            'chunks': chunks,
            'index': index,
            'embeddings': embeddings,
            'co': co
        })

        st.success("✅ Document ready!")

    # Query
    query = st.text_input("🔍 Ask a question")
    if query and 'index' in st.session_state:
        with st.spinner("🔎 Searching..."):
            q_embedding = embed_texts(st.session_state['co'], [query])
            if q_embedding.size == 0:
                st.error("❌ Query embedding failed.")
            else:
                D, I = st.session_state['index'].search(q_embedding, k=2)  # ⚡ fewer chunks
                relevant_chunks = [st.session_state['chunks'][idx] for idx in I[0] if 0 <= idx < len(st.session_state['chunks'])]

                answer = generate_answer(st.session_state['co'], relevant_chunks, query)

                st.success("✅ Answer:")
                st.write(answer)

                st.markdown("### 📖 Sources")
                for c in relevant_chunks:
                    st.markdown(
                        f'<div class="source-box"><b>Page {c["page"]}</b><br>{c["text"]}</div>',
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
