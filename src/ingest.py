from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pathlib import Path

import pdfplumber
import faiss
import os
import pickle

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text : str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

def build_faiss_index(embeddings, chunks, save_path="vectorstore"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(save_path, exist_ok=True)
    faiss.write_index(index, f"{save_path}/faiss.index")
    with open(f"{save_path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def pipeline(pdf_dir: str):
    all_chunks = []
    for pdf_file in Path(pdf_dir).glob("*.pdf"):
        print(f"Processing {pdf_file.name}")
        raw_text = extract_text_from_pdf(str(pdf_file))
        chunks = chunk_text(raw_text)
        all_chunks.extend(chunks)

    embeddings = embed_chunks(all_chunks)
    build_faiss_index(embeddings, all_chunks)

if __name__ == "__main__":
    pdf_dir = os.getenv("PDF_DIR", "data/raw_pdfs")
    pipeline(pdf_dir)