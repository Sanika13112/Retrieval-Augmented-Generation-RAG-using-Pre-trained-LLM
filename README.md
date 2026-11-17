# Academic RAG Assistant — Practical No. 06

A small Streamlit RAG project inspired by the workflow and structure of the reference repository:
https://github.com/AmrutaKanse09/RAG-using-Pre-trained-LLM

This is an independently written academic-domain adaptation. It uses:
- Engineering subject notes and research papers as the domain collection
- PyMuPDF for PDF extraction
- Word-based chunking
- BM25 (`rank-bm25`) for retrieval
- Groq API with a configurable pretrained LLM for answer generation
- Streamlit for the web interface
- Side-by-side RAG vs non-RAG answers, source passages, timing, and CSV experiment history

## 1. Project structure

```text
Academic_RAG_Streamlit_Project/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
└── documents/
    └── (put your reference PDF/TXT files here)
```

## 2. Requirements

- Python 3.10 or newer
- Internet connection
- Groq API key
- Text-based PDF or TXT documents

## 3. Setup on Windows / VS Code

Open this project folder in VS Code terminal.

Create a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install packages:

```powershell
pip install -r requirements.txt
```

## 4. Configure Groq API key

1. Create or sign in to a Groq account: https://console.groq.com/
2. Create an API key.
3. Copy `.env.example` to a new file named `.env`.
4. Put your key in `.env`:

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Do not share your API key or commit `.env` to GitHub. The `.gitignore` excludes it.

If the configured model is unavailable for your Groq account, use a currently supported model ID from your Groq console and update `GROQ_MODEL`.

## 5. Run the application

```powershell
streamlit run app.py
```

The app opens in your browser, usually at http://localhost:8501.

## 6. Add your documents

Option A: Upload PDF/TXT files using the sidebar file uploader. The selected category applies to that upload batch. Upload engineering notes and research papers in separate batches/runs if both categories are needed together.

Option B: Place PDF/TXT reference documents in the `documents/` folder. Local folder documents are labelled Engineering Notes by default in this starter version.

Recommended files:
- ML_Notes.pdf
- Deep_Learning_Notes.pdf
- RAG_Research_Paper.pdf
- NLP_Research_Paper.pdf

Use real documents you are allowed to use. Scanned/image-only PDFs may produce no extracted text unless OCR is added.

## 7. How it works

1. Extract text from PDF/TXT documents.
2. Split text into overlapping word chunks.
3. Tokenize each chunk.
4. BM25 ranks chunks against the question.
5. Send retrieved chunks + question to the LLM (RAG).
6. Send only the question to the same LLM (non-RAG).
7. Display both answers, retrieved passages, response times and experiment history.

## 8. Experiment

Test at least five questions:
- Explain supervised learning and its types.
- What is overfitting and how can it be reduced?
- What methodology is proposed in the research paper?
- Which dataset and metrics were used?
- What limitations or future work are discussed?

Evaluate both answers for correctness, relevance and completeness. Evaluate RAG grounding and citation quality. Do not conclude RAG is better based only on speed.

## Note on reference

The linked repository describes a real-estate RAG example using Streamlit, Groq, BM25, PyMuPDF and comparison of RAG/non-RAG answers. This project adapts that overall workflow to Computer Science/AIML engineering notes and research papers; it does not copy the original repository's code or real-estate content.
