from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import HUGGINGFACEHUB_API_TOKEN, EMBEDDING_MODEL_NAME
import os

class VectorStore:
    def __init__(self):
        if not HUGGINGFACEHUB_API_TOKEN:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
            
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self.persist_directory = "chroma_db"
        
        # Try to load existing vector store on initialization
        self.load_vector_store()

    def create_vector_store(self, documents):
        """Create a new vector store from documents."""
        if not documents:
            return False
            
        # Create a new ChromaDB instance with HNSW index
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 100,
                "hnsw:search_ef": 50
            }
        )
        # Save the vector store immediately after creation
        self.save_vector_store()
        return True

    def save_vector_store(self):
        """Save the vector store to disk."""
        if not self.vector_store:
            raise ValueError("No vector store to save")
        self.vector_store.persist()

    def load_vector_store(self):
        """Load the vector store from disk."""
        if os.path.exists(self.persist_directory):
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
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

    def get_collection_stats(self):
        """Get statistics about the vector store collection."""
        if not self.vector_store:
            return None
            
        try:
            collection = self.vector_store._collection
            return {
                'count': collection.count(),
                'metadata': collection.metadata
            }
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
            return None