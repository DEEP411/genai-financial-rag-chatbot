# Financial RAG Chatbot - AI-Powered Personal Finance Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.13-green)
![OpenAI](https://img.shields.io/badge/OpenAI-Latest-black)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-orange)
![RAG](https://img.shields.io/badge/Architecture-RAG-brightgreen)

An intelligent chatbot that answers questions about personal financial data using Retrieval Augmented Generation (RAG). Ask natural language questions about your spending patterns, categories, and financial health.

## Features

- **Semantic Search**: Finds relevant transactions using vector embeddings and FAISS
- **RAG Architecture**: Retrieves real transaction data before generating answers
- **LangChain Integration**: Uses LangChain for prompt engineering and chain orchestration
- **Conversational Interface**: Natural, multi-turn conversation with financial context
- **Prompt Engineering**: Sophisticated prompts with few-shot examples and structured outputs
- **Works Without API Key**: Demo mode shows RAG pipeline in action without OpenAI key
- **Type-Safe**: Full Python type hints for maintainability
- **Production-Ready**: Error handling, logging, and graceful degradation

## What is RAG?

**Retrieval Augmented Generation (RAG)** combines information retrieval with language models:

1. **Retrieval**: Search for relevant documents/data using semantic similarity
2. **Augmentation**: Include retrieved context in the prompt
3. **Generation**: Have the LLM generate answers based on actual data

This approach ensures answers are grounded in real financial data rather than relying solely on model training data.

## Tech Stack

```
Data Storage: CSV
Embeddings: OpenAI / HuggingFace
Vector DB: FAISS (Facebook AI Similarity Search)
LLM: GPT-3.5 Turbo (OpenAI)
Framework: LangChain
Language: Python 3.10+
```

## Project Structure

```
genai-financial-rag-chatbot/
├── app.py                     # Main application & CLI interface
├── rag_pipeline.py            # Core RAG pipeline implementation
├── prompt_templates.py        # Engineered prompts for financial analysis
├── requirements.txt           # Python dependencies
├── .env.example              # Environment configuration template
├── README.md                 # This file
└── data/
    └── sample_transactions.csv # Sample financial transaction data
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/DEEP411/genai-financial-rag-chatbot.git
cd genai-financial-rag-chatbot
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment (Optional)
For full functionality with OpenAI models:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-key-here
```

**Note**: The chatbot works in demo mode without an API key, showing the RAG pipeline in action.

## Usage

### Interactive Mode
```bash
python app.py
```

### Example Questions
```
"What are my top spending categories?"
"How much did I spend on groceries this month?"
"Show me my largest expenses"
"What's my spending pattern by category?"
"How much have I spent on transportation?"
```

## How It Works

```
User Question
     |
     v
[Vector Embedding]
     |
     v
[FAISS Vector Store] <-- Semantic Search
     |
     v
[Retrieved Transactions]
     |
     v
[LLM + Prompt Template + Context]
     |
     v
[Generated Answer]
```

## Author

**Deep Patel**
- GitHub: [github.com/DEEP411](https://github.com/DEEP411)
- Email: deepmpatel348@gmail.com
- LinkedIn: [linkedin.com/in/deeppatel348d](https://linkedin.com/in/deeppatel348d)

## License

MIT License
