from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)

    print("Total pages in PDF:", len(reader.pages))

    pages = []

    for page in reader.pages:
        text = page.extract_text()
        print(text)

        if text:
            pages.append(text)

    return pages


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    documents = load_pdf(pdf_path)

    print("Total pages extracted:", len(documents))