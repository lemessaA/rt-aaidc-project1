import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_documents() -> List[dict]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents with content and metadata
    """
    results = []
    data_dir = "data"
    
    # Read all .txt files from the data directory
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            results.append({
                                "content": content,
                                "metadata": {"source": filename, "type": "text"}
                            })
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    
    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    Includes session-based memory and logging capabilities.
    """

    def __init__(self, enable_memory: bool = True, enable_reasoning: bool = True):
        """
        Initialize the RAG assistant.
        
        Args:
            enable_memory: Enable conversation memory (default: True)
            enable_reasoning: Enable intermediate reasoning steps (default: True)
        """
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Initialize conversation memory
        self.enable_memory = enable_memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        ) if enable_memory else None

        # Enable reasoning steps
        self.enable_reasoning = enable_reasoning

        # Create RAG prompt template with memory support
        if enable_memory:
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant that answers questions based on the provided context.
You can reason through problems step by step when needed.

When answering:
1. First, analyze the question and the provided context
2. Identify the most relevant information
3. Reason through the answer step by step
4. Provide a clear, concise answer

If the answer cannot be found in the context, say so clearly."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", """Context information:
{context}

Question: {question}

Please provide your reasoning and answer:""")
            ])
        else:
            self.prompt_template = ChatPromptTemplate.from_template(
                """You are a helpful AI assistant that answers questions based on the provided context.

When answering, reason through the problem step by step:
1. Analyze the question and context
2. Identify relevant information
3. Reason through the answer
4. Provide a clear response

Context information:
{context}

Question: {question}

Reasoning and Answer:"""
            )

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        logger.info("RAG Assistant initialized successfully")
        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3, show_reasoning: Optional[bool] = None) -> Dict[str, Any]:
        """
        Query the RAG assistant with intermediate reasoning steps.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve
            show_reasoning: Override global reasoning setting (optional)

        Returns:
            Dictionary containing answer, reasoning steps, and metadata
        """
        show_reasoning = show_reasoning if show_reasoning is not None else self.enable_reasoning
        
        logger.info(f"Received query: {input}")
        
        # Step 1: Retrieve relevant context chunks from vector database
        logger.info(f"Step 1: Retrieving top {n_results} relevant chunks from vector database...")
        search_results = self.vector_db.search(input, n_results=n_results)
        
        # Log retrieval results
        retrieved_count = len(search_results.get("documents", []))
        logger.info(f"Retrieved {retrieved_count} document chunks")
        
        # Step 2: Combine retrieved document chunks into context
        context_parts = []
        documents = search_results.get("documents", [])
        if documents:
            for idx, doc in enumerate(documents):
                if doc and doc.strip():
                    context_parts.append(doc)
                    if show_reasoning:
                        logger.debug(f"Retrieved chunk {idx + 1}: {doc[:100]}...")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Step 3: Prepare prompt with memory if enabled
        if self.enable_memory and self.memory:
            chat_history = self.memory.chat_memory.messages
        else:
            chat_history = []
        
        # Step 4: Generate response using the RAG chain
        logger.info("Step 2: Generating response with LLM...")
        prompt_vars = {"context": context, "question": input}
        if self.enable_memory:
            prompt_vars["chat_history"] = chat_history
        
        llm_answer = self.chain.invoke(prompt_vars)
        
        # Step 5: Update memory with conversation
        if self.enable_memory and self.memory:
            self.memory.chat_memory.add_user_message(input)
            self.memory.chat_memory.add_ai_message(llm_answer)
        
        # Log the response
        logger.info(f"Generated response (length: {len(llm_answer)} chars)")
        
        # Prepare return value with reasoning steps
        result = {
            "answer": llm_answer,
            "query": input,
            "retrieved_chunks": retrieved_count,
            "timestamp": datetime.now().isoformat()
        }
        
        if show_reasoning:
            result["reasoning_steps"] = {
                "step_1": f"Retrieved {retrieved_count} relevant document chunks",
                "step_2": f"Combined context from {len(context_parts)} chunks",
                "step_3": "Generated response using LLM with context",
                "context_preview": context[:200] + "..." if len(context) > 200 else context
            }
        
        return result


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)

        done = False
        print("\n" + "="*60)
        print("RAG Assistant Ready!")
        print("="*60)
        print("Commands:")
        print("  - Type your question to get an answer")
        print("  - Type 'quit' or 'exit' to end the session")
        print("  - Type 'clear' to clear conversation memory")
        print("  - Type 'reasoning on/off' to toggle reasoning steps")
        print("="*60 + "\n")

        while not done:
            question = input("Enter a question or 'quit' to exit: ").strip()
            
            if question.lower() in ["quit", "exit"]:
                done = True
                logger.info("User ended session")
                print("\nThank you for using RAG Assistant. Goodbye!\n")
            elif question.lower() == "clear":
                if assistant.memory:
                    assistant.memory.clear()
                    logger.info("Conversation memory cleared")
                    print("\nConversation memory cleared.\n")
                else:
                    print("\nMemory is not enabled.\n")
            elif question.lower().startswith("reasoning"):
                parts = question.lower().split()
                if len(parts) > 1 and parts[1] in ["on", "off"]:
                    assistant.enable_reasoning = (parts[1] == "on")
                    logger.info(f"Reasoning steps {'enabled' if assistant.enable_reasoning else 'disabled'}")
                    print(f"\nReasoning steps {'enabled' if assistant.enable_reasoning else 'disabled'}.\n")
                else:
                    print("\nUsage: 'reasoning on' or 'reasoning off'\n")
            elif question:
                try:
                    result = assistant.invoke(question)
                    
                    # Display results
                    if isinstance(result, dict):
                        print("\n" + "-"*60)
                        if result.get("reasoning_steps") and assistant.enable_reasoning:
                            print("REASONING STEPS:")
                            for step, desc in result["reasoning_steps"].items():
                                if step != "context_preview":
                                    print(f"  {step.replace('_', ' ').title()}: {desc}")
                            print("-"*60)
                        
                        print(f"\nANSWER:")
                        print(result["answer"])
                        print("-"*60)
                        print(f"Retrieved {result['retrieved_chunks']} relevant chunks")
                        print("="*60 + "\n")
                    else:
                        # Fallback for string responses
                        print(f"\nAnswer: {result}\n")
                except Exception as e:
                    logger.error(f"Error processing query: {e}", exc_info=True)
                    print(f"\nError: {e}\n")
            else:
                print("Please enter a valid question.\n")

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()