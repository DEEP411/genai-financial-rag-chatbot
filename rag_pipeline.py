"""
RAG Pipeline for Financial Data Analysis.

This module implements the core Retrieval Augmented Generation (RAG) pipeline:
- Loads financial transaction CSV data
- Creates text embeddings using OpenAI or HuggingFace
- Stores embeddings in FAISS vector database
- Implements semantic search for relevant transactions
- Builds LangChain retrieval chains for Q&A

Key RAG concepts demonstrated:
- Document loading and preprocessing
- Text splitting for optimal chunking
- Semantic search using embeddings
- Vector store indexing
- Retrieval chain integration
"""

import os
from typing import List, Optional, Tuple
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()


def load_financial_data(csv_path: str) -> pd.DataFrame:
    """
    Load financial transaction data from CSV.

    Args:
        csv_path: Path to the CSV file containing transaction data

    Returns:
        DataFrame with financial transactions

    Raises:
        FileNotFoundError: If CSV file does not exist
        ValueError: If CSV is missing required columns
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Transaction data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = ['date', 'description', 'category', 'amount', 'type', 'balance']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"CSV missing required columns: {missing_columns}")

    return df


def prepare_transaction_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convert transaction dataframe to LangChain documents.

    Formats each transaction as a rich text document with metadata.

    Args:
        df: DataFrame with transaction data

    Returns:
        List of LangChain Document objects
    """
    documents = []

    for _, row in df.iterrows():
        content = f"""
Date: {row['date']}
Description: {row['description']}
Category: {row['category']}
Amount: ${abs(float(row['amount'])):.2f}
Type: {row['type']}
Balance: ${float(row['balance']):.2f}
"""
        metadata = {
            'date': str(row['date']),
            'category': str(row['category']),
            'amount': float(row['amount']),
            'type': str(row['type']),
            'description': str(row['description']),
        }

        doc = Document(page_content=content.strip(), metadata=metadata)
        documents.append(doc)

    return documents


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter for chunking transaction documents.

    Returns:
        RecursiveCharacterTextSplitter configured for financial data
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n", " ", ""]
    )


def create_embeddings(use_openai: bool = True):
    """
    Create an embeddings model.

    Args:
        use_openai: If True, use OpenAI embeddings. If False, use HuggingFace.

    Returns:
        Embeddings model instance

    Raises:
        ValueError: If OpenAI API key is missing and OpenAI embeddings requested
    """
    if use_openai:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in .env file or use "
                "create_embeddings(use_openai=False) for HuggingFace embeddings."
            )
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=api_key)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_vector_store(
    documents: List[Document],
    embeddings,
    store_path: str = "./faiss_store"
) -> FAISS:
    """
    Create and save a FAISS vector store from documents.

    Args:
        documents: List of Document objects to embed
        embeddings: Embeddings model to use
        store_path: Path to save the vector store

    Returns:
        FAISS vector store instance
    """
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(store_path)
    return vector_store


def load_vector_store(
    embeddings,
    store_path: str = "./faiss_store"
) -> Optional[FAISS]:
    """
    Load an existing FAISS vector store.

    Args:
        embeddings: Embeddings model to use
        store_path: Path to the saved vector store

    Returns:
        FAISS vector store if it exists, None otherwise
    """
    if not os.path.exists(store_path):
        return None

    try:
        return FAISS.load_local(store_path, embeddings)
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None


def build_rag_pipeline(csv_path: str, use_openai: bool = True) -> Tuple:
    """
    Build complete RAG pipeline from CSV to vector store.

    This is the main orchestrator function that:
    1. Loads financial data
    2. Prepares documents
    3. Creates embeddings
    4. Builds vector store

    Args:
        csv_path: Path to transaction CSV file
        use_openai: Whether to use OpenAI embeddings

    Returns:
        Tuple of (vector_store, documents, embeddings)
    """
    print(f"Loading financial data from {csv_path}...")
    df = load_financial_data(csv_path)
    print(f"Loaded {len(df)} transactions")

    print("Preparing transaction documents...")
    documents = prepare_transaction_documents(df)

    print("Creating embeddings...")
    try:
        embeddings = create_embeddings(use_openai=use_openai)
    except ValueError as e:
        print(f"OpenAI embeddings failed: {e}")
        print("Falling back to HuggingFace embeddings...")
        embeddings = create_embeddings(use_openai=False)

    print("Splitting and embedding documents...")
    splitter = create_text_splitter()
    split_docs = splitter.split_documents(documents)

    print(f"Creating FAISS vector store from {len(split_docs)} chunks...")
    vector_store = create_vector_store(split_docs, embeddings)

    return vector_store, documents, embeddings


def create_retriever(vector_store: FAISS, k: int = 5):
    """
    Create a retriever from the vector store.

    Args:
        vector_store: FAISS vector store instance
        k: Number of documents to retrieve

    Returns:
        Retriever instance
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def search_transactions(
    vector_store: FAISS,
    query: str,
    k: int = 5
) -> List[Tuple[Document, float]]:
    """
    Search for transactions similar to the query.

    Args:
        vector_store: FAISS vector store instance
        query: Search query
        k: Number of results to return

    Returns:
        List of (document, similarity_score) tuples
    """
    return vector_store.similarity_search_with_score(query, k=k)


def format_documents_for_context(documents: List[Document]) -> str:
    """
    Format retrieved documents into context string for LLM.

    Args:
        documents: List of retrieved documents

    Returns:
        Formatted string with document contents
    """
    if not documents:
        return "No relevant transactions found."

    formatted = []
    for i, doc in enumerate(documents, 1):
        formatted.append(f"Transaction {i}:\n{doc.page_content}")

    return "\n\n".join(formatted)
