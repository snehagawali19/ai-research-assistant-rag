from document_loader import load_pdf
from text_chunker import chunk_documents
from retriever import build_vector_index, search
from qa_system import generate_answer


pages = load_pdf("data/sample.pdf")
chunks = chunk_documents(pages)

index, texts, model = build_vector_index(chunks)

query = "What is attention mechanism?"

retrieved_chunks = search(query, index, texts, model)

answer = generate_answer(query, retrieved_chunks)

print("\nQuestion:", query)
print("\nAnswer:\n", answer)
def process_pdf(file_path):
    # your ingestion + chunking + embedding + FAISS storage
    pass

def query_rag(question):
    # your retrieval + LLM generation
    return "Your generated answer"