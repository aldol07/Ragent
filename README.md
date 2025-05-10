# RAG-Powered Multi-Agent Q&A Assistant

A sophisticated Question-Answering system that combines Retrieval-Augmented Generation (RAG) with multiple specialized agents for handling different types of queries. The system uses advanced language models and vector embeddings to provide accurate, context-aware responses.

## Architecture

### Core Components

1. **Document Processor**
   - Handles PDF document ingestion and processing
   - Splits documents into meaningful chunks
   - Prepares text for vector storage

2. **Vector Store**
   - Uses ChromaDB for efficient vector storage and retrieval
   - Implements similarity search with normalized scoring
   - Handles duplicate detection and unique content filtering
   - Persists embeddings for faster subsequent queries

3. **Agent System**
   - Multi-agent architecture for specialized query handling
   - Main components:
     - RAG Agent: Handles general queries using retrieved context
     - Calculator Agent: Processes mathematical expressions
     - Definition Agent: Provides clear definitions
     - Feedback Handler: Manages user feedback and acknowledgments

4. **Web Interface**
   - Streamlit-based interactive UI
   - Real-time document processing
   - Chat interface with context visualization
   - Similarity score display

### Key Design Choices

1. **Model Selection**
   - Uses Zephyr-7B-beta for high-quality responses
   - Implements HuggingFace embeddings for semantic search
   - Balances performance and resource requirements

2. **RAG Implementation**
   - Context-aware responses using retrieved information
   - Similarity threshold-based context selection
   - Normalized similarity scoring for better relevance ranking
   - Duplicate content filtering for diverse responses

3. **Response Processing**
   - Template-free response generation
   - Clean formatting and presentation
   - Context-aware answer generation
   - Fallback mechanisms for insufficient context

4. **Performance Optimizations**
   - Caching for frequently accessed components
   - Efficient vector storage and retrieval
   - Batch processing for document ingestion
   - Normalized similarity scores for better ranking

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- HuggingFace API token
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Create a .env file with:
   HUGGINGFACEHUB_API_TOKEN=your_token_here
   ```

### Running the Application

1. Place your PDF documents in the `docs` folder

2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Access the web interface at `http://localhost:8501`

4. Click "Process Documents" in the sidebar to index your documents

5. Start asking questions in the chat interface

### Deployment on Streamlit Cloud

1. Fork this repository to your GitHub account

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Create a new app and connect it to your forked repository

4. Set the following secrets in Streamlit Cloud:
   - `HUGGINGFACEHUB_API_TOKEN`: Your HuggingFace API token

5. Deploy the app

Note: The application uses ChromaDB with HNSW indexing for efficient vector storage and retrieval. This implementation is optimized for deployment on Streamlit Cloud and doesn't require additional system dependencies.

## Usage

1. **Document Processing**
   - Add PDF files to the `docs` folder
   - Click "Process Documents" to index them
   - View processing statistics in the sidebar

2. **Asking Questions**
   - Type your question in the chat input
   - View the response and retrieved context
   - Check similarity scores for context relevance

3. **Special Features**
   - Mathematical calculations (e.g., "calculate 2 + 2")
   - Definition requests (e.g., "define quantum computing")
   - General knowledge questions
   - Context-aware responses

## Project Structure

```
.
├── app.py                 # Streamlit web application
├── agent.py              # Multi-agent system implementation
├── vector_store.py       # Vector storage and retrieval
├── document_processor.py # Document processing and chunking
├── config.py            # Configuration and constants
├── requirements.txt     # Project dependencies
├── docs/               # Document storage
└── chroma_db/         # Vector store persistence
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## Acknowledgments

- HuggingFace for the language models and embeddings
- ChromaDB for vector storage
- Streamlit for the web interface 