import streamlit as st
from document_processor import DocumentProcessor
from vector_store import VectorStore
from agent import Agent
import os

# Initialize components
@st.cache_resource
def initialize_components():
    try:
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()
        agent = Agent()
        return doc_processor, vector_store, agent
    except ValueError as e:
        st.error(f"Initialization error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error during initialization: {str(e)}")
        st.stop()

def main():
    st.title("RAG-Powered Multi-Agent Q&A Assistant")

    # Initialize components
    try:
        doc_processor, vector_store, agent = initialize_components()
    except Exception as e:
        st.error("Failed to initialize components. Please check your environment variables and try again.")
        st.stop()

    # Sidebar for document processing
    with st.sidebar:
        st.header("Document Processing")
        if st.button("Process Documents"):
            try:
                with st.spinner("Processing documents..."):
                    # Process documents
                    chunks = doc_processor.process_documents("docs")
                    if not chunks:
                        st.warning("No documents were processed. Please check if there are PDF files in the docs folder.")
                        return

                    # Create and save vector store
                    vector_store.create_vector_store(chunks)
                    vector_store.save_vector_store()

                    # Display collection stats
                    stats = vector_store.get_collection_stats()
                    if stats:
                        st.success(f"Documents processed and indexed successfully!")
                        st.info(f"Collection stats: {stats['count']} chunks, {stats['dimensions']} dimensions")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

    # Main chat interface
    st.header("Ask a Question")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "context" in message and message["context"]:
                with st.expander("View Retrieved Context"):
                    for i, chunk in enumerate(message["context_chunks"]):
                        st.write(f"**Chunk {i+1}**")
                        st.write(chunk)
                        if "similarity_scores" in message and i < len(message["similarity_scores"]):
                            st.write(message["similarity_scores"][i])
                        st.write("---")

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Process the query
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    # Get relevant context
                    relevant_docs = vector_store.similarity_search(prompt)
                    
                    # Format context and collect similarity scores
                    context_parts = []
                    context_chunks = []
                    similarity_scores = []
                    for doc in relevant_docs:
                        context_parts.append(doc['content'])
                        context_chunks.append(doc['content'])
                        similarity_scores.append(doc['similarity_score'])
                    context = "\n\n".join(context_parts)
                    
                    # Set similarity threshold
                    similarity_threshold = 0.5
                    # Process query with agent, passing similarity scores
                    result = agent.process_query(prompt, context, similarity_scores, similarity_threshold)
                    
                    # Display response
                    st.write(result["response"])
                    
                    # Display decision and context
                    st.info(f"Decision: {result['decision']}")
                    
                    if context:
                        with st.expander("View Retrieved Context"):
                            for i, chunk in enumerate(context_chunks):
                                st.write(f"**Chunk {i+1}** (Relevance: {similarity_scores[i]:.2%})")
                                st.write(chunk)
                                st.write("---")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "context": context if context else None,
                        "context_chunks": context_chunks if context_chunks else None,
                        "similarity_scores": [f"{score:.2%}" for score in similarity_scores] if similarity_scores else None
                    })
            except Exception as e:
                st.error(f"Error processing your query: {str(e)}")

if __name__ == "__main__":
    main()
