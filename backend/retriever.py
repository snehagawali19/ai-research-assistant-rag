import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from document_loader import load_pdf
from text_chunker import chunk_documents


def build_vector_index(chunks):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk.page_content for chunk in chunks]

    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index, texts, model


def search(query, index, texts, model, top_k=2):

    query_vector = model.encode([query])

    distances, indices = index.search(query_vector, top_k)

    results = []

    for idx in indices[0]:
        results.append(texts[idx])

    return results


if __name__ == "__main__":

    pages = load_pdf("data/sample.pdf")

    chunks = chunk_documents(pages)

    index, texts, model = build_vector_index(chunks)

    query = "What is attention mechanism?"

    results = search(query, index, texts, model)

    print("\nUser Question:")
    print(query)

    print("\nTop Relevant Chunks:\n")

    for i, r in enumerate(results):
        print(f"Result {i+1}:")
        print(r[:300])
        print()