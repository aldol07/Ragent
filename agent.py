from langchain_community.llms import HuggingFaceHub
from config import HUGGINGFACEHUB_API_TOKEN, CALCULATOR_KEYWORDS, DEFINE_KEYWORDS
import re
import os

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class Agent:
    def __init__(self):
        if not HUGGINGFACEHUB_API_TOKEN:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
        
        # Initialize the Zephyr model with newer LangChain format
        self.model = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            model_kwargs={
                "temperature": 0.7,
                "max_length": 512,
                "return_full_text": False
            }
        )
        
        self.conversation_history = []
    
    def _is_calculator_query(self, query):
        """Check if the query requires calculator."""
        return any(keyword in query.lower() for keyword in CALCULATOR_KEYWORDS)

    def _is_definition_query(self, query):
        """Check if the query requires definition lookup."""
        return any(keyword in query.lower() for keyword in DEFINE_KEYWORDS)

    def _extract_math_expression(self, query):
        """Extract mathematical expression from query, supporting negative numbers and parentheses."""
        # Try to extract everything after a calculator keyword
        for keyword in CALCULATOR_KEYWORDS:
            if keyword in query.lower():
                # Extract everything after the keyword
                idx = query.lower().find(keyword) + len(keyword)
                expr = query[idx:].strip()
                if expr:
                    return expr
        # Fallback: regex for numbers, operators, parentheses, and spaces
        pattern = r'([\d\s\+\-\*\/\(\)]+)'
        match = re.search(pattern, query)
        if match:
            return match.group(1).strip()
        return None

    def _calculate(self, expression):
        """Safely evaluate mathematical expression."""
        try:
            return eval(expression)
        except:
            return "Invalid mathematical expression"

    def _clean_response(self, response):
        """Clean up the response by removing prompt templates and system messages."""
        if isinstance(response, str):
            # Remove common prompt templates and system messages
            templates_to_remove = [
                "Answer the following question as a helpful assistant:",
                "Based on the following context, please answer the question.",
                "If the context doesn't contain enough information, say so.",
                "Context:",
                "Question:",
                "Answer:",
                "You are a helpful assistant.",
                "Please provide a clear and concise answer.",
                "Here's the answer:",
                "Here's what I found:",
                "Based on the context:",
                "According to the context:"
            ]
            
            for template in templates_to_remove:
                response = response.replace(template, "")
            
            # Remove any leading/trailing whitespace and newlines
            response = response.strip()
            
            # Remove any remaining prompt-like patterns
            response = re.sub(r'^(Here|Based|According|The|A|An)\s+', '', response)
            response = re.sub(r'\n+', '\n', response)  # Replace multiple newlines with single newline
            
        return response

    def process_query(self, query, context=None, similarity_scores=None, similarity_threshold=0.5):
        """Process the query and determine the appropriate action."""
        if not query:
            return {"decision": "Error", "response": "No query provided"}
            
        decision = "RAG"
        response = None
        
        try:
            # Check if it's a general statement or feedback
            if any(word in query.lower() for word in ["thank", "thanks", "good", "great", "awesome", "nice", "excellent"]):
                decision = "Feedback"
                response = "Thank you for your feedback! I'm glad I could help."
                return {"decision": decision, "response": response}

            # Check query type and route accordingly
            if self._is_calculator_query(query):
                decision = "Calculator"
                expression = self._extract_math_expression(query)
                if expression:
                    result = self._calculate(expression)
                    response = f"The result of {expression} is {result}"
                else:
                    response = "I couldn't identify a valid mathematical expression in your query."
                    
            elif self._is_definition_query(query):
                decision = "Definition"
                # Use Zephyr for definitions
                prompt = f"Provide a clear and concise definition for: {query}"
                response = self._clean_response(self.model.invoke(prompt))
                
            else:
                # RAG pipeline with similarity threshold
                use_context = False
                if context and similarity_scores:
                    # Check if any score meets the threshold
                    use_context = any(score >= similarity_threshold for score in similarity_scores)
                    
                if use_context:
                    # Create prompt without showing it in output
                    system_prompt = "You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain enough information, say so."
                    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                    response = self._clean_response(self.model.invoke(prompt))
                else:
                    # Fallback to general LLM answering
                    prompt = f"Answer the following question as a helpful assistant: {query}"
                    response = self._clean_response(self.model.invoke(prompt))
                    decision = "LLM"
                
        except Exception as e:
            decision = "Error"
            response = f"An error occurred while processing your query: {str(e)}"
            
        # Log the interaction
        self.conversation_history.append({
            "query": query,
            "decision": decision,
            "response": response
        })
        
        return {
            "decision": decision,
            "response": response
        }
