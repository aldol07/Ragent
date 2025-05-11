# RAG-Powered Multi-Agent Q&A Assistant

A sophisticated Question-Answering system that combines Retrieval-Augmented Generation (RAG) with multiple specialized agents for handling different types of queries. The system uses advanced language models and vector embeddings to provide accurate, context-aware responses.

🔗 **Live Demo**: [ragent-07.streamlit.app](https://ragent-07.streamlit.app/)  
📂 **GitHub Repo**: [github.com/aldol07/Ragent](https://github.com/aldol07/Ragent)

---

## 🧠 Architecture

### 🔧 Core Components

1. **Document Processor**
   - Ingests and preprocesses PDF documents
   - Splits content into contextually meaningful chunks
   - Prepares text for vector embedding

2. **Vector Store (FAISS)**
   - Utilizes FAISS for fast, scalable vector similarity search
   - Embeddings powered by `sentence-transformers/all-MiniLM-L6-v2`
   - Handles duplicate filtering and persistent storage
   - Optimized for both local and cloud deployment

3. **Agent System**
   - Modular multi-agent architecture with specialized query handlers:
     - **RAG Agent**: Answers context-aware general queries
     - **Calculator Agent**: Solves mathematical expressions
     - **Definition Agent**: Responds to "define ..." queries
     - **Feedback Handler**: Handles acknowledgments and feedback

4. **Web Interface**
   - Built with Streamlit for fast, interactive UI
   - Sidebar controls for document upload and indexing
   - Chat interface with live context visualization
   - Displays similarity scores alongside answers

---

### 🎯 Key Design Choices

- **Embeddings**: Implements `sentence-transformers/all-MiniLM-L6-v2` for semantic retrieval
- **RAG Logic**:
  - Retrieves context based on similarity threshold
  - Normalized scoring for better relevance
  - Removes redundant chunks before answering
- **Optimizations**:
  - Persistent FAISS index for quick restarts
  - Caching of embeddings and agent responses
  - Batch chunking and filtering pipeline

---

## ⚙️ Setup Instructions

### ✅ Prerequisites

- Python 3.8+
- HuggingFace API token
- `faiss-cpu` or `faiss-gpu` (depending on system)
- Dependencies in `requirements.txt`

### 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/aldol07/Ragent.git
cd Ragent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

▶️ Running the Application
Place your PDF files into the docs/ directory

Start the Streamlit app:

streamlit run app.py
Visit http://localhost:8501 in your browser

Click "Process Documents" in the sidebar

Start asking questions in the chat interface

☁️ Deploying on Streamlit Cloud
Fork this repository to your GitHub account

Go to Streamlit Cloud

Create a new app and link to your forked repo

Add your HuggingFace token in Secrets:
HUGGINGFACEHUB_API_TOKEN=your_token_here
Ensure faiss-cpu is listed in requirements.txt

Deploy the app 🚀

💬 Usage Guide
Document Processing
Add one or more .pdf files to the docs/ folder

Click "Process Documents" in the sidebar to start indexing

Interacting with the Assistant
Ask general or domain-specific questions

View retrieved context, similarity scores, and full answers

Try:

"define reinforcement learning"

"calculate (25 + 33) * 2"

"explain the plot of Macbeth"

Features
Multi-agent architecture

Context-aware RAG answers

Math solving and definition lookups

User feedback management

📁 Project Structure
graphql
Copy
Edit
.
├── app.py                 # Streamlit web application
├── agent.py               # Multi-agent system logic
├── vector_store.py        # FAISS-based vector store
├── document_processor.py  # PDF chunking and preprocessing
├── config.py              # Configurations and constants
├── requirements.txt       # Python dependencies
├── docs/                  # Input PDF documents
└── faiss_index/           # Persistent FAISS index (auto-created)
🤝 Contributing
Fork the repository

Create a new feature branch

Commit and push your changes

Open a Pull Request for review


🙏 Acknowledgments
HuggingFace — for language models & embeddings

FAISS — for fast vector search

Streamlit — for an elegant and reactive UI

Langchain — for utility chains and agent tooling
