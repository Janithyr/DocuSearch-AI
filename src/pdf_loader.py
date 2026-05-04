import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    doc.close()
    return full_text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def load_and_chunk_pdf(file_path):
    print(f"Loading PDF: {file_path}")
    text = load_pdf(file_path)
    print(f"Total characters extracted: {len(text)}")

    chunks = split_text(text)
    print(f"Total chunks created: {len(chunks)}")

    return chunks