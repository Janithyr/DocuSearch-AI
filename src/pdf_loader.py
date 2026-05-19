import fitz


def load_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    doc.close()
    return full_text


def split_text(text, chunk_size=500, chunk_overlap=50):
    if not text:
        return []

    words = " ".join(text.split()).split(" ")
    chunks = []
    current = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 > chunk_size and current:
            chunk = " ".join(current)
            chunks.append(chunk)

            overlap_words = []
            overlap_len = 0
            for w in reversed(current):
                overlap_len += len(w) + 1
                overlap_words.append(w)
                if overlap_len >= chunk_overlap:
                    break
            overlap_words.reverse()

            current = overlap_words
            current_len = sum(len(w) for w in current) + max(len(current) - 1, 0)

        current.append(word)
        current_len += len(word) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def load_and_chunk_pdf(file_path):
    print(f"Loading PDF: {file_path}")
    text = load_pdf(file_path)
    print(f"Total characters extracted: {len(text)}")

    chunks = split_text(text)
    print(f"Total chunks created: {len(chunks)}")

    return chunks