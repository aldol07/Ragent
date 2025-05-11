from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import HUGGINGFACEHUB_API_TOKEN
import os
import numpy as np

class VectorStore:
    def __init__(self):
        if not HUGGINGFACEHUB_API_TOKEN:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
            
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.persist_directory = "vector_store"
        self.vector_store = None
        self.load_vector_store()

    def create_vector_store(self, documents):
        """Create a new vector store from documents."""
        if not documents:
            return False
            
        try:
            # Create a new vector store directly from documents
            self.vector_store = FAISS.from_documents(
                documents,
                self.embeddings
            )
            
            # Save the vector store immediately after creation
            self.save_vector_store()
            return True
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return False

    def save_vector_store(self):
        """Save the vector store to disk."""
        if not self.vector_store:
            raise ValueError("No vector store to save")
        try:
            self.vector_store.save_local(self.persist_directory)
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
            raise

    def load_vector_store(self):
        """Load the vector store from disk."""
        if os.path.exists(self.persist_directory):
            try:
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings
                )
                return True
            except Exception as e:
                print(f"Error loading vector store: {str(e)}")
                return False
        return False

    def similarity_search(self, query):
        """Perform similarity search on the vector store and return top 3 relevant chunks."""
        if not self.vector_store:
            return []
            
        try:
            # Get more results initially to ensure we have enough unique chunks
            results = self.vector_store.similarity_search_with_score(
                query,
                k=6  # Get more results to ensure we have enough unique chunks
            )
            
            # Format results to include normalized similarity scores
            formatted_results = []
            seen_contents = set()
            
            if results:
                # Get the highest score for normalization
                max_score = max(score for _, score in results)
                min_score = min(score for _, score in results)
                
                # Normalize scores to be between 0 and 1
                for doc, score in results:
                    # Skip if we've already seen this content
                    if doc.page_content in seen_contents:
                        continue
                        
                    # Add to seen contents
                    seen_contents.add(doc.page_content)
                    
                    # Normalize score to be between 0 and 1
                    if max_score != min_score:
                        normalized_score = (score - min_score) / (max_score - min_score)
                    else:
                        normalized_score = 1.0  # If all scores are the same
                    
                    # Add slight variation to identical scores to ensure uniqueness
                    if len(formatted_results) > 0 and normalized_score == formatted_results[-1]['similarity_score']:
                        normalized_score = max(0.0, normalized_score - 0.01)
                        
                    formatted_results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': normalized_score
                    })
                    
                    # Stop if we have 3 unique results
                    if len(formatted_results) >= 3:
                        break
                
                # If we don't have 3 unique results, pad with the last result
                while len(formatted_results) < 3 and len(results) > 0:
                    last_doc, last_score = results[-1]
                    formatted_results.append({
                        'content': last_doc.page_content,
                        'metadata': last_doc.metadata,
                        'similarity_score': 0.0  # Set a low score for padded results
                    })
                
            return formatted_results
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []

    def get_collection_stats(self):
        """Get statistics about the vector store collection."""
        if not self.vector_store:
            return None
            
        try:
            return {
                'count': len(self.vector_store.index_to_docstore_id),
                'dimensions': self.dimension,
                'metadata': {'type': 'FAISS'}
            }
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
            return None