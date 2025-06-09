import faiss
import pickle
from sentence_transformers import SentenceTransformer

def load_index(path="vectorstore"):
    index = faiss.read_index(f"{path}/faiss.index")
    with open(f"{path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, model

def query_rag_index(query: str, top_k=5):
    index, chunks, model = load_index()
    q_embedding = model.encode([query])
    D, I = index.search(q_embedding, top_k)
    return [chunks[i] for i in I[0]]

if __name__ == "__main__":
    query = input("Ask a question: ")
    results = query_rag_index(query)
    print("\nTop Matches:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r[:500]}\n{'-'*60}")
