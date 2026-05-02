"""
Financial RAG Chatbot - Main Application.

A CLI-based chatbot that uses Retrieval Augmented Generation (RAG) to answer
questions about personal financial data. The system:

1. Loads transaction data from CSV
2. Creates embeddings and stores them in FAISS vector database
3. Uses LangChain to build a conversational retrieval chain
4. Answers financial questions by retrieving relevant transactions
5. Provides analysis, insights, and recommendations

Features:
- Semantic search across financial transactions
- LLM-powered analysis and insights
- Multi-turn conversation support
- Graceful fallback when API keys missing
- Type hints and comprehensive documentation
"""

import os
import sys
from typing import Optional, Tuple
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

from rag_pipeline import (
    build_rag_pipeline,
    create_retriever,
    search_transactions,
    format_documents_for_context,
)
from prompt_templates import (
    get_financial_analysis_prompt,
    get_spending_summary_prompt,
)


load_dotenv()


class FinancialChatbot:
    """
    RAG-based chatbot for financial data analysis.

    This class orchestrates the RAG pipeline and provides a user-friendly
    interface for asking questions about financial transactions.
    """

    def __init__(self, csv_path: str = "data/sample_transactions.csv"):
        """
        Initialize the Financial Chatbot.

        Args:
            csv_path: Path to the transaction CSV file
        """
        self.csv_path = csv_path
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        self.memory = None
        self.api_key_available = False

    def initialize(self) -> bool:
        """
        Initialize the RAG pipeline and LLM.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print("\n" + "="*60)
            print("Financial RAG Chatbot - Initializing")
            print("="*60)

            api_key = os.getenv('OPENAI_API_KEY')
            use_openai = bool(api_key)

            if not use_openai:
                print("\nWARNING: OPENAI_API_KEY not found in .env file")
                print("Using demonstration mode with mock responses")
                print("Set OPENAI_API_KEY in .env to enable full functionality\n")
            else:
                self.api_key_available = True
                print("\nOpenAI API key detected. Full chatbot mode enabled.\n")

            print("Building RAG pipeline from financial data...")
            self.vector_store, _, _ = build_rag_pipeline(
                self.csv_path,
                use_openai=use_openai
            )

            if self.vector_store is None:
                print("ERROR: Failed to build vector store")
                return False

            self.retriever = create_retriever(self.vector_store, k=5)

            if self.api_key_available:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    api_key=api_key
                )
            else:
                self.llm = self._get_demo_llm()

            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            self._setup_qa_chain()

            print("Chatbot initialized successfully!")
            print("\nType 'help' for commands, 'exit' to quit.\n")
            return True

        except Exception as e:
            print(f"\nERROR during initialization: {e}")
            print("Please ensure the CSV file exists at", self.csv_path)
            return False

    def _get_demo_llm(self):
        """
        Get a demo LLM that provides canned responses.

        This allows the chatbot to work without an API key for demonstration.

        Returns:
            A mock LLM that simulates responses
        """
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerBase

        class DemoLLM(LLM):
            """Mock LLM for demonstration without API key."""

            @property
            def _llm_type(self) -> str:
                return "demo"

            def _call(self, prompt: str, **kwargs) -> str:
                return (
                    "Based on the retrieved transaction data, I can see relevant "
                    "financial information. However, to provide detailed analysis and "
                    "insights, please set up your OPENAI_API_KEY in the .env file. "
                    "This demo shows the RAG pipeline is working and can retrieve "
                    "transactions semantically related to your question."
                )

        return DemoLLM()

    def _setup_qa_chain(self) -> None:
        """Set up the RetrievalQA chain with financial analysis prompt."""
        prompt = get_financial_analysis_prompt()

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def query(self, question: str) -> dict:
        """
        Ask a question about financial data.

        Args:
            question: The question to ask

        Returns:
            Dictionary with 'answer' and 'source_documents' keys
        """
        if not self.qa_chain:
            return {
                "answer": "Chatbot not initialized. Call initialize() first.",
                "source_documents": []
            }

        try:
            result = self.qa_chain({"query": question})
            return result
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "source_documents": []
            }

    def interactive_mode(self) -> None:
        """Run the chatbot in interactive CLI mode."""
        print("\n" + "="*60)
        print("Financial RAG Chatbot - Interactive Mode")
        print("="*60)
        print("\nExample questions:")
        print("  - What are my largest expenses?")
        print("  - How much did I spend on groceries?")
        print("  - What's my spending pattern?")
        print("  - Which category has the most transactions?")
        print("  - How much have I spent this month?")
        print("\nCommands: 'help' - show this message, 'exit' - quit\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'exit':
                    print("\nThank you for using Financial RAG Chatbot. Goodbye!")
                    break

                if user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  exit - quit the chatbot")
                    print("  help - show this message")
                    print("\nAsk any question about your financial data!")
                    continue

                print("\nProcessing your question...")
                result = self.query(user_input)

                print("\nBot:", result["answer"])

                if result.get("source_documents"):
                    print("\n[Retrieved", len(result["source_documents"]),
                          "relevant transactions]")
                print()

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")

    def batch_queries(self, queries: list) -> None:
        """
        Run multiple queries in batch mode.

        Args:
            queries: List of questions to ask
        """
        print("\n" + "="*60)
        print("Financial RAG Chatbot - Batch Mode")
        print("="*60 + "\n")

        for i, query_text in enumerate(queries, 1):
            print(f"Query {i}: {query_text}")
            result = self.query(query_text)
            print(f"Answer: {result['answer']}\n")


def main() -> None:
    """Main entry point for the application."""
    csv_path = "data/sample_transactions.csv"

    if not os.path.exists(csv_path):
        print(f"ERROR: Transaction data not found at {csv_path}")
        print("Please ensure the 'data' folder contains 'sample_transactions.csv'")
        sys.exit(1)

    chatbot = FinancialChatbot(csv_path)

    if not chatbot.initialize():
        sys.exit(1)

    example_queries = [
        "What are my top spending categories?",
        "How much have I spent on dining this month?",
        "What are my transportation expenses?",
        "Show me my salary deposits",
    ]

    print("\nRunning example queries...\n")
    chatbot.batch_queries(example_queries)

    print("\n" + "="*60)
    print("Starting interactive mode...")
    print("="*60)
    chatbot.interactive_mode()


if __name__ == "__main__":
    main()
