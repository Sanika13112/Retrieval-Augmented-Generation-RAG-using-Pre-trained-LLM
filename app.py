import os
import re
import time
from pathlib import Path

import fitz
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi

# -------------------------------------------------
# 1. APPLICATION CONFIGURATION
# -------------------------------------------------
load_dotenv()
st.set_page_config(
    page_title="Academic RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Academic RAG Assistant")
st.caption("Engineering Notes & Research Papers | RAG vs LLM without retrieval")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Groq API key not found. Add GROQ_API_KEY to your .env file.")
    st.info("See README.md for setup instructions.")
    st.stop()

client = Groq(api_key=api_key)
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# -------------------------------------------------
# 2. TEXT EXTRACTION
# -------------------------------------------------
def extract_uploaded_text(uploaded_file):
    filename = uploaded_file.name
    lower = filename.lower()

    if lower.endswith(".pdf"):
        pdf = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        pages = []
        for page_no, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "name": filename,
                    "page": page_no,
                    "text": text
                })
        return pages

    if lower.endswith(".txt"):
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        return [{"name": filename, "page": 1, "text": text}] if text.strip() else []

    return []

def load_local_documents(folder="documents"):
    documents = []
    folder_path = Path(folder)
    folder_path.mkdir(exist_ok=True)

    for path in folder_path.iterdir():
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".txt":
                text = path.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    documents.append({
                        "name": path.name,
                        "page": 1,
                        "text": text
                    })
            elif path.suffix.lower() == ".pdf":
                pdf = fitz.open(path)
                for page_no, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        documents.append({
                            "name": path.name,
                            "page": page_no,
                            "text": text
                        })
        except Exception as exc:
            st.warning(f"Could not read {path.name}: {exc}")
    return documents

# -------------------------------------------------
# 3. DOCUMENT COLLECTION + METADATA
# -------------------------------------------------
def assign_category(filename, category_map):
    return category_map.get(filename, "Engineering Notes")

def collect_documents(uploaded_files, category_map):
    documents = load_local_documents()

    for file in uploaded_files or []:
        for page in extract_uploaded_text(file):
            page["category"] = assign_category(file.name, category_map)
            documents.append(page)

    # Remove duplicate (same filename/page) entries if a file is also in documents/.
    unique = {}
    for doc in documents:
        key = (doc["name"], doc["page"])
        unique[key] = doc

    return list(unique.values())

# -------------------------------------------------
# 4. TEXT CHUNKING
# -------------------------------------------------
def chunk_text(text, chunk_size=800, overlap=100):
    words = re.findall(r"\S+", text)
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks

def prepare_chunks(documents):
    all_chunks = []

    for doc in documents:
        for chunk_id, chunk in enumerate(chunk_text(doc["text"]), start=1):
            all_chunks.append({
                "source": doc["name"],
                "category": doc.get("category", "Engineering Notes"),
                "page": doc["page"],
                "chunk_id": chunk_id,
                "text": chunk
            })

    return all_chunks

# -------------------------------------------------
# 5. BM25 RETRIEVAL
# -------------------------------------------------
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def retrieve_documents(question, chunks, top_k=5, category_filter="All"):
    if not chunks:
        return []

    selected = [
        chunk for chunk in chunks
        if category_filter == "All" or chunk["category"] == category_filter
    ]
    if not selected:
        return []

    tokenized_chunks = [tokenize(chunk["text"]) for chunk in selected]
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenize(question))

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []
    for idx in ranked_indices[:top_k]:
        if scores[idx] > 0:
            result = selected[idx].copy()
            result["score"] = float(scores[idx])
            results.append(result)

    return results

# -------------------------------------------------
# 6. LLM RESPONSE GENERATION
# -------------------------------------------------
def generate_answer(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful academic question-answering assistant "
                    "for Computer Science and AIML engineering topics. "
                    "Give clear, structured and accurate answers."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# -------------------------------------------------
# 7. RAG ANSWER
# -------------------------------------------------
def rag_answer(question, retrieved_chunks):
    if not retrieved_chunks:
        return (
            "No relevant passages were retrieved from the selected documents. "
            "Try another question or upload relevant notes."
        )

    context = ""
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context += (
            f"\n[Source {i}: {chunk['source']} | "
            f"Category: {chunk['category']} | Page: {chunk['page']}]\n"
            f"{chunk['text']}\n"
        )

    prompt = f"""
Answer the question using the supplied document context.

Rules:
- Prioritize the supplied context.
- If the answer is not present in the context, clearly state that the uploaded
  documents do not provide enough information.
- Do not invent facts, experimental results, datasets or citations.
- Cite supporting sources using the filename and page number.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return generate_answer(prompt)

# -------------------------------------------------
# 8. NON-RAG ANSWER
# -------------------------------------------------
def non_rag_answer(question):
    prompt = f"""
Answer the following Computer Science / AIML question using your pretrained
knowledge only. You are NOT provided with any external documents.
Do not claim that you have read the user's uploaded notes or research papers.

QUESTION:
{question}

ANSWER:
"""
    return generate_answer(prompt)

# -------------------------------------------------
# 9. SIDEBAR: DOCUMENT UPLOAD + SETTINGS
# -------------------------------------------------
st.sidebar.header("📁 Document Collection")
uploaded_files = st.sidebar.file_uploader(
    "Upload engineering notes or research papers",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

st.sidebar.markdown("**Category for uploaded files**")
category_choice = st.sidebar.radio(
    "Choose category",
    ["Engineering Notes", "Research Paper"],
    help="Upload one category at a time. You can upload the other category in another run."
)

top_k = st.sidebar.slider(
    "Number of retrieved chunks",
    min_value=1,
    max_value=10,
    value=5
)

st.sidebar.info(
    f"Retrieval: BM25 keyword ranking\n\nLLM: {MODEL} via Groq"
)

# -------------------------------------------------
# 10. LOAD AND PROCESS DOCUMENTS
# -------------------------------------------------
category_map = {
    file.name: category_choice
    for file in (uploaded_files or [])
}

documents = collect_documents(uploaded_files, category_map)
chunks = prepare_chunks(documents)

st.subheader("📚 Document Collection")
col1, col2, col3 = st.columns(3)
col1.metric("Documents", len(set(d["name"] for d in documents)))
col2.metric("Extracted pages", len(documents))
col3.metric("Text chunks", len(chunks))

if documents:
    summary = pd.DataFrame([
        {
            "Document": d["name"],
            "Category": d.get("category", "Engineering Notes"),
            "Page": d["page"]
        }
        for d in documents
    ]).drop_duplicates()
    st.dataframe(summary, use_container_width=True, hide_index=True)
else:
    st.warning(
        "Upload PDF/TXT files in the sidebar or place PDF/TXT files in the "
        "'documents' folder."
    )

# -------------------------------------------------
# 11. USER QUESTION
# -------------------------------------------------
st.subheader("💬 Ask an Academic Question")
question = st.text_area(
    "Enter your question:",
    placeholder="Example: Explain the methodology used in the uploaded research paper.",
    height=100
)

category_filter = st.selectbox(
    "Search in document collection",
    ["All", "Engineering Notes", "Research Paper"]
)

compare = st.button(
    "🔍 Compare RAG and Non-RAG Answers",
    type="primary",
    use_container_width=True
)

# -------------------------------------------------
# 12. ANSWER COMPARISON
# -------------------------------------------------
if compare:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not chunks:
        st.error("No readable document text is available. Upload a text-based PDF/TXT.")
    else:
        with st.spinner("Retrieving passages and generating both answers..."):
            retrieval_start = time.perf_counter()
            retrieved = retrieve_documents(
                question, chunks, top_k, category_filter
            )
            retrieval_time = time.perf_counter() - retrieval_start

            rag_start = time.perf_counter()
            answer_with_rag = rag_answer(question, retrieved)
            rag_generation_time = time.perf_counter() - rag_start

            non_rag_start = time.perf_counter()
            answer_without_rag = non_rag_answer(question)
            non_rag_time = time.perf_counter() - non_rag_start

        st.success("Both answers generated successfully!")

        left, right = st.columns(2)
        with left:
            st.markdown("### 📖 Answer WITH RAG")
            st.markdown(answer_with_rag)
            st.caption(
                f"Retrieval: {retrieval_time:.4f}s | "
                f"LLM generation: {rag_generation_time:.2f}s"
            )

        with right:
            st.markdown("### 🧠 Answer WITHOUT RAG")
            st.markdown(answer_without_rag)
            st.caption(f"LLM generation: {non_rag_time:.2f}s")

        st.markdown("---")
        st.markdown("### 📌 Retrieved Source Passages")

        if retrieved:
            for i, item in enumerate(retrieved, start=1):
                with st.expander(
                    f"{i}. {item['source']} | {item['category']} | "
                    f"Page {item['page']} | BM25 score {item['score']:.3f}"
                ):
                    st.write(item["text"])
        else:
            st.info("BM25 did not find passages with matching query terms.")

        st.markdown("### ⏱️ Response Time Comparison")
        timing_df = pd.DataFrame({
            "Method": ["With RAG (retrieval + generation)", "Without RAG (generation)"],
            "Time (seconds)": [
                retrieval_time + rag_generation_time,
                non_rag_time
            ]
        })
        st.bar_chart(timing_df.set_index("Method"))

        # Save a record for local experimental analysis.
        if "experiment_log" not in st.session_state:
            st.session_state.experiment_log = []

        st.session_state.experiment_log.append({
            "Question": question,
            "RAG answer": answer_with_rag,
            "Non-RAG answer": answer_without_rag,
            "Retrieved chunks": len(retrieved),
            "Retrieval time (s)": round(retrieval_time, 4),
            "RAG generation time (s)": round(rag_generation_time, 3),
            "Non-RAG time (s)": round(non_rag_time, 3)
        })

if "experiment_log" in st.session_state and st.session_state.experiment_log:
    st.markdown("---")
    st.subheader("📊 Experiment History")
    history_df = pd.DataFrame(st.session_state.experiment_log)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download experiment results (CSV)",
        data=csv,
        file_name="rag_experiment_results.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption(
    "Made with ❤️ by Sanika | "
    
)
