from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import os

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def load_pdf(self, pdf_path):
        """Load and extract text from a PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {str(e)}")
            return ""

    def process_documents(self, docs_dir):
        """Process all PDF documents in the specified directory."""
        all_texts = []
        document_metadata = []
        
        for filename in os.listdir(docs_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(docs_dir, filename)
                text = self.load_pdf(pdf_path)
                if text:
                    all_texts.append(text)
                    document_metadata.append({"source": filename})
        
        # Split texts into chunks with metadata
        chunks = []
        for i, text in enumerate(all_texts):
            text_chunks = self.text_splitter.create_documents(
                [text], 
                metadatas=[document_metadata[i]]
            )
            chunks.extend(text_chunks)
        
        return chunks
