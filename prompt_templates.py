"""
Prompt templates for financial analysis and RAG chain.

This module contains carefully engineered prompts for:
- Financial analysis and insights
- Spending summaries
- Budget recommendations
- Transaction categorization

Key techniques demonstrated:
- System context and role definition
- Few-shot examples for better outputs
- Output formatting specifications
- Clear instructions for financial reasoning
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate


def get_financial_analysis_prompt() -> ChatPromptTemplate:
    """
    Create a prompt for analyzing financial transactions and answering questions.

    This prompt includes:
    - Clear system role for the AI
    - Context about the financial data
    - Instructions for analysis
    - Output formatting requirements

    Returns:
        ChatPromptTemplate: Formatted prompt for financial analysis
    """
    system_prompt = """You are an expert financial analyst assistant. Your role is to help users understand their spending patterns, financial health, and make informed financial decisions.

When analyzing transactions:
1. Look for patterns and trends in spending
2. Identify high-cost categories and outliers
3. Consider seasonal variations when mentioned
4. Be specific with numbers and percentages
5. Provide actionable insights

Always:
- Be honest about what you can and cannot determine from the data
- Use the actual transaction data provided, not assumptions
- Focus on facts from the financial records
- Give clear, concise explanations"""

    human_prompt = """Based on the provided financial transaction data, please answer the following question:

{question}

Relevant transaction records:
{context}

Provide a detailed, data-driven response:"""

    messages = [
        ("system", system_prompt),
        ("human", human_prompt),
    ]

    return ChatPromptTemplate.from_messages(messages)


def get_spending_summary_prompt() -> ChatPromptTemplate:
    """
    Create a prompt for generating spending summaries.

    This prompt helps summarize spending patterns across categories.

    Returns:
        ChatPromptTemplate: Formatted prompt for spending summary
    """
    system_prompt = """You are a personal finance advisor creating spending summaries.
Focus on clarity and actionable insights.
Format your response with clear sections and bullet points."""

    human_prompt = """Analyze these transaction records and create a spending summary:

{context}

Provide:
1. Total spending by category
2. Top spending categories
3. Average transaction value
4. Key observations about spending patterns"""

    messages = [
        ("system", system_prompt),
        ("human", human_prompt),
    ]

    return ChatPromptTemplate.from_messages(messages)


def get_budget_recommendation_prompt() -> ChatPromptTemplate:
    """
    Create a prompt for budget recommendations.

    This prompt generates personalized budget advice based on spending data.
    Uses few-shot examples to guide the AI's reasoning.

    Returns:
        ChatPromptTemplate: Formatted prompt for budget recommendations
    """
    examples = [
        {
            "spending": "High entertainment spending ($500/month), moderate groceries ($300/month)",
            "recommendation": "Consider setting a 30% limit on entertainment relative to discretionary income. Groceries appear reasonable."
        },
        {
            "spending": "Transportation $200/month, utilities $150/month, rent $1500/month",
            "recommendation": "Housing costs at reasonable 40-45% of typical income. Transportation is moderate. Utilities are efficient."
        }
    ]

    example_prompt = PromptTemplate(
        input_variables=["spending", "recommendation"],
        template="Spending Pattern: {spending}\nRecommendation: {recommendation}"
    )

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="""Based on the transaction data provided, analyze the spending pattern and recommend budget adjustments:

Transaction Summary:
{context}

Provide specific, actionable budget recommendations:""",
        input_variables=["context"],
    )

    return prompt


def get_category_analysis_prompt() -> ChatPromptTemplate:
    """
    Create a prompt for analyzing spending by category.

    Returns:
        ChatPromptTemplate: Formatted prompt for category analysis
    """
    system_prompt = """You are a financial analyst specializing in spending categorization and analysis.
Provide detailed breakdowns of spending by category.
Include percentages, trends, and comparisons."""

    human_prompt = """Analyze the spending by category from these transactions:

{context}

For each major category, provide:
- Total and average spending
- Percentage of total spending
- Notable patterns or anomalies
- Recommendations for optimization"""

    messages = [
        ("system", system_prompt),
        ("human", human_prompt),
    ]

    return ChatPromptTemplate.from_messages(messages)


def get_question_routing_prompt() -> ChatPromptTemplate:
    """
    Create a prompt for routing questions to appropriate analysis type.

    This helps the system determine what type of analysis is most relevant.

    Returns:
        ChatPromptTemplate: Formatted prompt for question routing
    """
    system_prompt = """You are a financial question classifier.
Determine what type of financial analysis would best answer the user's question."""

    human_prompt = """Classify this financial question into one category:

Question: {question}

Categories:
1. spending_summary - for questions about overall spending patterns
2. budget_recommendation - for questions about budgeting advice
3. category_analysis - for questions about specific spending categories
4. transaction_lookup - for questions about specific transactions
5. general_analysis - for general financial questions

Respond with just the category name."""

    messages = [
        ("system", system_prompt),
        ("human", human_prompt),
    ]

    return ChatPromptTemplate.from_messages(messages)
