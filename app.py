import streamlit as st
from document_processor import DocumentProcessor
from vector_store import VectorStore
from agent import Agent
import os
import traceback

@st.cache_resource
def initialize_components():
    try:
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()
        agent = Agent(vector_store=vector_store)
        return doc_processor, vector_store, agent
    except ValueError as e:
        st.error(f"Initialization error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error during initialization: {str(e)}")
        st.error(traceback.format_exc())
        st.stop()

def main():
    st.title("RAG-Powered Multi-Agent Q&A Assistant")

    try:
        doc_processor, vector_store, agent = initialize_components()
    except Exception as e:
        st.error("Failed to initialize components. Please check your environment variables and try again.")
        st.stop()

    vector_stats = vector_store.get_collection_stats()
    has_documents = vector_stats is not None and vector_stats.get('count', 0) > 0
    
    with st.sidebar:
        st.header("Document Processing")
        if has_documents:
            st.success(f"✅ Documents already loaded: {vector_stats['count']} chunks")
        else:
            st.warning("⚠️ No documents loaded yet")
            
        if st.button("Process Documents"):
            try:
                with st.spinner("Processing documents..."):
                    chunks = doc_processor.process_documents("docs")
                    if not chunks:
                        st.warning("No documents were processed. Please check if there are PDF files in the docs folder.")
                        return

                    success = vector_store.create_vector_store(chunks)
                    if not success:
                        st.error("Failed to create vector store. Please check the logs for details.")
                        return
                        
                    agent.update_vector_store(vector_store)
                    
                    stats = vector_store.get_collection_stats()
                    if stats:
                        st.success(f"Documents processed and indexed successfully!")
                        st.info(f"Collection stats: {stats['count']} chunks, {stats['dimensions']} dimensions")
                        st.rerun()
                    else:
                        st.warning("Could not retrieve collection statistics.")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                st.error("Please check if your documents are valid PDF files and try again.")
                st.error(traceback.format_exc())

    with st.sidebar:
        st.divider()
        debug_mode = st.checkbox("Debug Mode", value=False)

    st.header("Ask a Question")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "context" in message and message["context"] and debug_mode:
                with st.expander("View Retrieved Context"):
                    for i, chunk in enumerate(message["context_chunks"]):
                        st.write(f"**Chunk {i+1}**")
                        st.write(chunk)
                        if "similarity_scores" in message and i < len(message["similarity_scores"]):
                            st.write(message["similarity_scores"][i])
                        st.write("---")

    if prompt := st.chat_input("What would you like to know?"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

 
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    response = agent.answer_query(prompt)
                    
                    st.write(response["response"])
                    
                    if debug_mode:
                        st.info(f"Decision: {response['decision']}")
                    
                    if debug_mode and response.get('context_chunks'):
                        with st.expander("View Retrieved Context"):
                            for i, chunk in enumerate(response.get('context_chunks', [])):
                                score_str = f"(Relevance: {response.get('similarity_scores', [])[i]:.2%})" if i < len(response.get('similarity_scores', [])) else ""
                                st.write(f"**Chunk {i+1}** {score_str}")
                                st.write(chunk)
                                st.write("---")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["response"],
                        "context": response.get("context"),
                        "context_chunks": response.get("context_chunks"),
                        "similarity_scores": response.get("similarity_scores", [])
                    })
            except Exception as e:
                st.error(f"Error processing your query: {str(e)}")
                if debug_mode:
                    st.error(traceback.format_exc())
                st.error("Please try again or rephrase your question.")

if __name__ == "__main__":
    main()