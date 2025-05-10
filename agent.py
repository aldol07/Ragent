from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
        
        # Create prompt templates
        self.definition_template = PromptTemplate(
            input_variables=["query"],
            template="""You are a helpful assistant. Provide a clear, concise, and accurate definition for the following term or concept. 
            Include relevant examples if helpful, but keep the response focused and informative.

            Term/Concept: {query}

            Definition:"""
        )
        
        self.rag_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""You are a helpful assistant. Answer the question based on the provided context. 
            If the context doesn't contain enough information, say so and provide a general answer based on your knowledge.
            Make your response clear, concise, and well-structured.

            Context:
            {context}

            Question: {query}

            Answer:"""
        )
        
        self.general_template = PromptTemplate(
            input_variables=["query"],
            template="""You are a helpful assistant. Provide a clear, informative, and well-structured answer to the following question.
            If you're not completely sure about something, acknowledge that and provide the best information you have.

            Question: {query}

            Answer:"""
        )
        
        # Create chains
        self.definition_chain = LLMChain(llm=self.model, prompt=self.definition_template)
        self.rag_chain = LLMChain(llm=self.model, prompt=self.rag_template)
        self.general_chain = LLMChain(llm=self.model, prompt=self.general_template)
        
        self.conversation_history = []
    
    def _is_calculator_query(self, query):
        """Check if the query requires calculator."""
        # Check for mathematical operators
        math_operators = ['+', '-', '*', '/', '÷', '×', '^', '**']
        return any(op in query for op in math_operators) or any(keyword in query.lower() for keyword in CALCULATOR_KEYWORDS)

    def _is_definition_query(self, query):
        """Check if the query requires definition lookup."""
        return any(keyword in query.lower() for keyword in DEFINE_KEYWORDS)

    def _extract_math_expression(self, query):
        """Extract mathematical expression from query."""
        # Remove any text before or after the expression
        pattern = r'([\d\s\+\-\*\/\(\)\^]+)'
        match = re.search(pattern, query)
        if match:
            expr = match.group(1).strip()
            # Replace common math symbols
            expr = expr.replace('×', '*').replace('÷', '/').replace('^', '**')
            return expr
        return None

    def _calculate(self, expression):
        """Safely evaluate mathematical expression."""
        try:
            # Follow order of operations: parentheses, exponents, multiplication/division, addition/subtraction
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating expression: {str(e)}"

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
                "According to the context:",
                "Definition:",
                "Term/Concept:"
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
                    response = self._calculate(expression)
                else:
                    response = "I couldn't identify a valid mathematical expression in your query."
                    
            elif self._is_definition_query(query):
                decision = "Definition"
                # Use definition chain
                response = self._clean_response(self.definition_chain.invoke({"query": query})["text"])
                
            else:
                # RAG pipeline with similarity threshold
                use_context = False
                if context and similarity_scores:
                    # Check if any score meets the threshold
                    use_context = any(score >= similarity_threshold for score in similarity_scores)
                    
                if use_context:
                    # Use RAG chain
                    response = self._clean_response(self.rag_chain.invoke({
                        "context": context,
                        "query": query
                    })["text"])
                else:
                    # Fallback to general chain
                    response = self._clean_response(self.general_chain.invoke({"query": query})["text"])
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
