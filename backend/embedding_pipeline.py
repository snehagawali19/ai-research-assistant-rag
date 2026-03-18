from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from text_chunker import chunk_documents
from document_loader import load_pdf


def create_embeddings(chunks):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk.page_content for chunk in chunks]

    embeddings = model.encode(texts)

    return embeddings, texts


def store_vectors(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    faiss.write_index(index, "vector_db/index.faiss")

    return index


if __name__ == "__main__":

    pages = load_pdf("data/sample.pdf")

    chunks = chunk_documents(pages)

    embeddings, texts = create_embeddings(chunks)

    index = store_vectors(np.array(embeddings))

    print("Total chunks:", len(texts))
    print("Embedding dimension:", embeddings.shape)
    print("Vectors stored in FAISS index.")