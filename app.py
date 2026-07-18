import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os

st.set_page_config(page_title="Multi-PDF Chatbot", page_icon="📚", layout="wide")

# --- Load models once, cached across reruns ---
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

embed_model = load_embed_model()
client = load_groq_client()

# --- Session state ---
if "doc_cache" not in st.session_state:
    st.session_state.doc_cache = {}

def process_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    if len(text.strip()) < 20:
        return None, "⚠️ Couldn't extract readable text (possibly a scanned PDF)."
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    embeddings = embed_model.encode(chunks).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return {"chunks": chunks, "index": index}, None

def retrieve_balanced(query, per_doc_k=2):
    query_embedding = embed_model.encode([query]).astype('float32')
    results = []
    for fname, data in st.session_state.doc_cache.items():
        k = min(per_doc_k, len(data["chunks"]))
        distances, indices = data["index"].search(query_embedding, k)
        for i in indices[0]:
            results.append((data["chunks"][i], fname))
    return results

def ask_question(query, per_doc_k=2):
    retrieved = retrieve_balanced(query, per_doc_k)
    context = "\n\n".join([f"[From {src}]: {chunk}" for chunk, src in retrieved])
    
    prompt = f"""Answer the question based only on the context below. Mention which document(s) the answer comes from if relevant.
If the answer isn't in the context, say "I couldn't find that in the document(s)." Do not make up information.

Context:
{context}

Question: {query}

Answer:"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content, retrieved

# --- UI ---
st.title("📚 Multi-PDF Chatbot")
st.caption("Upload PDFs and ask questions across all of them.")

with st.sidebar:
    st.header("📂 Documents")
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.doc_cache:
                with st.spinner(f"Processing {f.name}..."):
                    data, error = process_pdf(f)
                    if error:
                        st.error(f"{f.name}: {error}")
                    else:
                        st.session_state.doc_cache[f.name] = data
    
    if st.session_state.doc_cache:
        st.write("**Loaded documents:**")
        for fname in list(st.session_state.doc_cache.keys()):
            col1, col2 = st.columns([4, 1])
            col1.write(f"✅ {fname} ({len(st.session_state.doc_cache[fname]['chunks'])} chunks)")
            if col2.button("✕", key=f"del_{fname}"):
                del st.session_state.doc_cache[fname]
                st.rerun()
    else:
        st.info("No documents uploaded yet.")

# --- Chat interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask a question about your documents...")

if query:
    if not st.session_state.doc_cache:
        st.warning("Please upload at least one PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = ask_question(query)
                st.write(answer)
                with st.expander("📄 Source chunks used"):
                    for i, (chunk, src) in enumerate(sources):
                        st.markdown(f"**[{i+1}] from {src}**")
                        st.text(chunk[:300] + "...")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
