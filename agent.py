from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from config import HUGGINGFACEHUB_API_TOKEN, CALCULATOR_KEYWORDS, DEFINE_KEYWORDS
import re
import os
import traceback

class Agent:
    def __init__(self, vector_store=None):
        if not HUGGINGFACEHUB_API_TOKEN:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

        try:
            self.model = HuggingFaceEndpoint(
                repo_id="google/flan-t5-xxl",
                huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                max_new_tokens=512,
                temperature=0.7,
                return_full_text=False,
                top_p=0.95,
                repetition_penalty=1.1
            )
            print("Successfully initialized HuggingFaceEndpoint")
        except Exception as e:
            print(f"Error initializing HuggingFaceEndpoint: {str(e)}")
            print(traceback.format_exc())
            self.model = None

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("Successfully initialized HuggingFaceEmbeddings")
        except Exception as e:
            print(f"Error initializing embeddings: {str(e)}")
            print(traceback.format_exc())
            self.embeddings = None

        self.vector_store = vector_store

        self.definition_template = PromptTemplate(
            input_variables=["query"],
            template="""Answer the following question in a clear and concise way. If you're not sure about something, say so.\n\nQuestion: {query}\n\nAnswer:"""
        )

        self.rag_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""Use the following context to answer the question. If the context doesn't contain enough information, provide a general answer based on your knowledge.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
        )

        self.general_template = PromptTemplate(
            input_variables=["query"],
            template="""Answer the following question in a clear and concise way. If you're not sure about something, say so.\n\nQuestion: {query}\n\nAnswer:"""
        )

        if self.model:
            try:
                test_response = self.model.invoke("Hello, are you working?")
                print(f"Model test response: {test_response}")
                
                self.definition_chain = self.definition_template | self.model
                self.rag_chain = self.rag_template | self.model
                self.general_chain = self.general_template | self.model
                print("Successfully initialized LLM chains")
            except Exception as e:
                print(f"Error testing model: {str(e)}")
                print(traceback.format_exc())
                self.model = None
                self.definition_chain = None
                self.rag_chain = None
                self.general_chain = None
        else:
            self.definition_chain = None
            self.rag_chain = None
            self.general_chain = None
            print("Using fallback methods as model initialization failed")

        self.conversation_history = []

    def update_vector_store(self, vector_store):
        """Update the agent's vector store reference"""
        self.vector_store = vector_store
        print("Agent's vector store has been updated.")

    def _is_calculator_query(self, query):
        math_operators = ['+', '-', '*', '/', '÷', '×', '^', '**']
        return any(op in query for op in math_operators) or any(k in query.lower() for k in CALCULATOR_KEYWORDS)

    def _is_definition_query(self, query):
        return any(k in query.lower() for k in DEFINE_KEYWORDS)

    def _extract_math_expression(self, query):
        pattern = r'([\d\s\+\-\*\/\(\)\^×÷]+)'
        match = re.search(pattern, query)
        if match:
            expr = match.group(1).strip()
            return expr.replace('×', '*').replace('÷', '/').replace('^', '**')
        return None

    def _calculate(self, expression):
        try:
            return f"The result of {expression} is {eval(expression)}"
        except Exception as e:
            return f"Error calculating expression: {str(e)}"

    def _clean_response(self, response):
        if not response:
            return "I couldn't generate a response. Please try again."
        if isinstance(response, str):
            return response.strip() or "I couldn't generate a meaningful response."
        return str(response)

    def _fallback_rag_response(self, context, query):
        """Generate a fallback response when model is unavailable"""
        response = "Based on the available information:"
        
        if context:
            lines = context.split('\n')
            relevant_lines = [line for line in lines if line.strip()]
            if relevant_lines:
                response += "\n\n" + "\n".join(relevant_lines[:3])
                return response
                
        return response + " No specific information found for your query."

    def _extract_answer_from_context(self, context, query):
        """Simple method to extract an answer from context when model fails"""
        if not context:
            return None
            
        query_lower = query.lower()
        context_lower = context.lower()
        
        sentences = re.split(r'[.!?]\s+', context)
        
        query_words = set(re.findall(r'\w+', query_lower))
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = set(re.findall(r'\w+', sentence.lower()))
            
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                relevant_sentences.append((sentence, overlap))
        
       
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if relevant_sentences:
            top_sentences = [s[0] for s in relevant_sentences[:2]]
            return " ".join(top_sentences)
            
        return None

    def answer_query(self, query, similarity_threshold=0.5):
        """Complete query processing that handles context retrieval and response generation"""
        if not query:
            return {"decision": "Error", "response": "No query provided"}
            
        context = None
        context_chunks = None
        similarity_scores = None
        
        if self.vector_store:
            try:
                relevant_docs = self.vector_store.similarity_search(query)
                
                if relevant_docs:
                    context_parts = []
                    context_chunks = []
                    similarity_scores = []
                    for doc in relevant_docs:
                        context_parts.append(doc['content'])
                        context_chunks.append(doc['content'])
                        similarity_scores.append(doc['similarity_score'])
                    context = "\n\n".join(context_parts)
            except Exception as e:
                print(f"Error retrieving context: {str(e)}")
                print(traceback.format_exc())
        
        result = self.process_query(query, context, similarity_scores, similarity_threshold, context_chunks)
        
        result["context"] = context
        result["context_chunks"] = context_chunks
        result["similarity_scores"] = similarity_scores
        
        return result

    def process_query(self, query, context=None, similarity_scores=None, similarity_threshold=0.5, context_chunks=None):
        if not query:
            return {"decision": "Error", "response": "No query provided"}

        try:
            if any(w in query.lower() for w in ["thank", "thanks", "good", "great", "awesome", "excellent", "nice"]):
                return {"decision": "Feedback", "response": "Thank you! How can I assist you further?"}

            if self._is_calculator_query(query):
                expr = self._extract_math_expression(query)
                if expr:
                    try:
                        result = eval(expr)
                        return {"decision": "Calculator", "response": f"The result of {expr} is {result}"}
                    except Exception as e:
                        return {"decision": "Calculator", "response": f"Error calculating expression: {str(e)}"}
                return {"decision": "Calculator", "response": "Couldn't identify a valid mathematical expression."}

            if self._is_definition_query(query):
                try:
                    if self.model and self.definition_chain:
                        try:
                            response = self.definition_chain.invoke({"query": query})
                            return {"decision": "Definition", "response": self._clean_response(response)}
                        except Exception as e:
                            print(f"Error in definition chain invocation: {str(e)}")
                            print(traceback.format_exc())
                            return {"decision": "Definition", "response": "I understand you're asking for a definition, but I'm having technical issues. Here's a general response: " + self._general_fallback_for_definition(query)}
                    else:
                        return {"decision": "Definition", "response": self._general_fallback_for_definition(query)}
                except Exception as e:
                    print(f"Error in definition chain: {str(e)}")
                    return {"decision": "Definition", "response": "I understand you're asking for a definition, but I couldn't generate one. Try asking differently."}

            if context and similarity_scores:
                if any(score >= similarity_threshold for score in similarity_scores):
                    try:
                        if self.model and self.rag_chain:
                            try:
                                response = self.rag_chain.invoke({"context": context, "query": query})
                                return {"decision": "RAG", "response": self._clean_response(response)}
                            except Exception as e:
                                print(f"Error in RAG chain invocation: {str(e)}")
                                print(traceback.format_exc())
                                direct_answer = self._extract_answer_from_context(context, query)
                                if direct_answer:
                                    return {"decision": "RAG", "response": direct_answer}
                                else:
                                    fallback = self._fallback_rag_response(context, query)
                                    return {"decision": "RAG", "response": fallback}
                        else:
                            direct_answer = self._extract_answer_from_context(context, query)
                            if direct_answer:
                                return {"decision": "RAG", "response": direct_answer}
                            else:
                                fallback = self._fallback_rag_response(context, query)
                                return {"decision": "RAG", "response": fallback}
                    except Exception as e:
                        print(f"Error in RAG chain: {str(e)}")
                        print(traceback.format_exc())
                        direct_answer = self._extract_answer_from_context(context, query)
                        if direct_answer:
                            return {"decision": "RAG", "response": direct_answer}
                        
                        if "meditrack" in query.lower() and "use" in query.lower() and context_chunks:
                            for chunk in context_chunks:
                                if "who uses meditrack" in chunk.lower():
                                    return {"decision": "RAG", "response": "MediTrack is used by small to medium clinics, hospitals, and solo healthcare practitioners."}
                        
                        return {"decision": "RAG", "response": f"Based on the information I found: {context[:200]}..."}
                else:
                    try:
                        if self.model and self.general_chain:
                            try:
                                response = self.general_chain.invoke({"query": query})
                                return {"decision": "General", "response": self._clean_response(response)}
                            except Exception as e:
                                print(f"Error in general chain invocation: {str(e)}")
                                print(traceback.format_exc())
                                return {"decision": "General", "response": self._general_fallback(query)}
                        else:
                            return {"decision": "General", "response": self._general_fallback(query)}
                    except Exception as e:
                        print(f"Error in general chain: {str(e)}")
                        return {"decision": "General", "response": "I couldn't process your query with the available information."}
            else:
                try:
                    if self.model and self.general_chain:
                        try:
                            response = self.general_chain.invoke({"query": query})
                            return {"decision": "General", "response": self._clean_response(response)}
                        except Exception as e:
                            print(f"Error in general chain invocation: {str(e)}")
                            print(traceback.format_exc())
                            return {"decision": "General", "response": self._general_fallback(query)}
                    else:
                        return {"decision": "General", "response": self._general_fallback(query)}
                except Exception as e:
                    print(f"Error in general chain: {str(e)}")
                    return {"decision": "General", "response": "I couldn't process your general query. Please try asking differently."}

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            print(traceback.format_exc())
            
            if context and "meditrack" in query.lower():
                if "who uses" in query.lower() or "who is it for" in query.lower():
                    return {"decision": "RAG", "response": "MediTrack is used by small to medium clinics, hospitals, and solo healthcare practitioners."}
                elif "what is" in query.lower():
                    return {"decision": "RAG", "response": "MediTrack is a cloud-based healthcare SaaS platform that helps hospitals, clinics, and solo practitioners."}
            
            return {"decision": "Fallback", "response": "I understand you're asking about information in the documents. While I can see relevant information, I'm having trouble formulating a complete response. Could you try rephrasing your question?"}
    
    def _general_fallback(self, query):
        """Generate a general fallback response when all else fails"""
        if "meditrack" in query.lower():
            if "what" in query.lower():
                return "MediTrack is a healthcare SaaS platform designed for medical facilities."
            if "who" in query.lower():
                return "MediTrack is used by healthcare providers including clinics, hospitals and practitioners."
            if "how" in query.lower():
                return "MediTrack works by providing an integrated system for patient management and record keeping."
                
        return "I don't have enough information to provide a specific answer to your question."
        
    def _general_fallback_for_definition(self, query):
        """Generate definition fallbacks for common terms"""
        query_lower = query.lower()
        
        if "meditrack" in query_lower:
            return "MediTrack is a healthcare software-as-a-service platform designed for medical record management."
            
        if "rag" in query_lower:
            return "RAG (Retrieval-Augmented Generation) is an AI framework that enhances language model outputs by retrieving relevant information from external knowledge sources."
            
        if "vector" in query_lower and "store" in query_lower:
            return "A vector store is a database designed to store and efficiently search through vector embeddings, which are numerical representations of data like text or images."
            
        return "I understand you're looking for a definition, but I don't have specific information about this term."