from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_pdf

def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.create_documents(pages)
    return chunks


if __name__ == "__main__":
    pages = load_pdf("data/sample.pdf")

    print("Total pages in PDF:", len(pages))

    chunks = chunk_documents(pages)

    print("Total chunks created:", len(chunks))

    if len(chunks) > 0:
        print("\nFirst chunk preview:\n")
        print(chunks[0].page_content[:300])
    else:
        print("No chunks created. The PDF may not contain extractable text.")