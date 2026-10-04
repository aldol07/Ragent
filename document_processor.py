import io
import os
import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class Chunk:
    """A piece of a document plus where it came from (used for citations)."""
    text: str
    source: str
    page: int  # 1-based
    chunk_id: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self):
        return f"{self.source}, p. {self.page}"


def _clean(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class DocumentProcessor:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        )

    def extract_pages(self, name, data):
        """Return [(page_number, text)] for a PDF given as bytes."""
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = _clean(page.extract_text() or "")
            except Exception as e:  # a single broken page should not sink the file
                print(f"[ingest] {name} p.{i}: extraction failed ({e})")
                text = ""
            if text:
                pages.append((i, text))
        return pages

    def chunk_pdf(self, name, data, start_id=0):
        chunks = []
        for page_no, text in self.extract_pages(name, data):
            for piece in self.splitter.split_text(text):
                piece = piece.strip()
                if len(piece) < 20:
                    continue
                chunks.append(Chunk(text=piece, source=name, page=page_no, chunk_id=start_id + len(chunks)))
        return chunks

    def process_files(self, files):
        """files: iterable of (filename, bytes). Returns (chunks, per-file report)."""
        chunks, report = [], {}
        for name, data in files:
            name = os.path.basename(name)
            try:
                file_chunks = self.chunk_pdf(name, data, start_id=len(chunks))
                chunks.extend(file_chunks)
                report[name] = len(file_chunks)
            except Exception as e:
                print(f"[ingest] {name}: failed ({e})")
                report[name] = 0
        return chunks, report

    def process_directory(self, docs_dir):
        files = []
        for fname in sorted(os.listdir(docs_dir)):
            if fname.lower().endswith(".pdf"):
                with open(os.path.join(docs_dir, fname), "rb") as f:
                    files.append((fname, f.read()))
        return self.process_files(files)
