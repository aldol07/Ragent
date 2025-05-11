import os
from dotenv import load_dotenv
load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
TOP_K_RESULTS = 3

CALCULATOR_KEYWORDS = ["calculate", "compute", "sum", "multiply", "divide", "add"]
DEFINE_KEYWORDS = ["define", "what is", "meaning of", "definition of"] 