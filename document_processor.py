from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import os
import traceback
from config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def process_documents(self, docs_dir):
        """Process PDF documents from the specified directory and return chunks."""
        print("\n=== Document Processing Status ===")

        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
            print(f"Created {docs_dir} directory. Please add your PDF documents there.")
            return []

        try:
            pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]

            if not pdf_files:
                print("No PDF files found in the docs directory.")
                return []

            print(f"\nFound {len(pdf_files)} PDF file(s):")
            for pdf in pdf_files:
                print(f"- {pdf}")

            documents = []

            for pdf_file in pdf_files:
                try:
                    file_path = os.path.join(docs_dir, pdf_file)
                    print(f"\nProcessing {pdf_file}...")
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    if docs:
                        documents.extend(docs)
                        print(f"✓ Successfully loaded {pdf_file}")
                    else:
                        print(f"⚠ Warning: No content in {pdf_file}")
                except Exception as e:
                    print(f"✗ Error loading {pdf_file}: {str(e)}")
                    print("Traceback:", traceback.format_exc())

            if not documents:
                print("No content extracted from PDFs.")
                return []

            chunks = self.text_splitter.split_documents(documents)

            if not chunks:
                print("Text splitting resulted in no chunks.")
                return []
                
            print(f"\n✓ Document processing complete: {len(chunks)} chunks created")
            return chunks

        except Exception as e:
            print(f"\n✗ Error processing documents: {str(e)}")
            print("Traceback:", traceback.format_exc())
            return []