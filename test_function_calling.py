import os
import pandas as pd
from dotenv import load_dotenv
from google import genai

from tools.analysis_tools import (
    dataset_summary,
    total_sales,
    total_profit,
    total_quantity,
    profit_margin,
    sales_by_region,
    sales_by_product,
    profit_by_region,
    profit_by_product,
    monthly_sales_profit
)

load_dotenv()


# Load dataset
df = pd.read_csv("data/sample/sample.csv")


# --------------------------------------------------
# TOOL WRAPPERS
# --------------------------------------------------

def get_dataset_summary():
    """Get the basic structure and quality information of the dataset."""
    return dataset_summary(df)


def get_total_sales():
    """Calculate the total sales."""
    return total_sales(df)


def get_total_profit():
    """Calculate the total profit."""
    return total_profit(df)


def get_total_quantity():
    """Calculate the total quantity sold."""
    return total_quantity(df)


def get_profit_margin():
    """Calculate the overall profit margin percentage."""
    return profit_margin(df)


def get_sales_by_region():
    """Calculate total sales for each region."""
    return sales_by_region(df)


def get_sales_by_product():
    """Calculate total sales for each product."""
    return sales_by_product(df)


def get_profit_by_region():
    """Calculate total profit for each region."""
    return profit_by_region(df)


def get_profit_by_product():
    """Calculate total profit for each product."""
    return profit_by_product(df)


def get_monthly_sales_profit():
    """Calculate monthly sales and profit."""
    return monthly_sales_profit(df)


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# --------------------------------------------------
# CREATE CHAT WITH TOOLS
# --------------------------------------------------

chat = client.chats.create(
    model="models/gemini-3-flash-preview",
    config={
        "tools": [
            get_dataset_summary,
            get_total_sales,
            get_total_profit,
            get_total_quantity,
            get_profit_margin,
            get_sales_by_region,
            get_sales_by_product,
            get_profit_by_region,
            get_profit_by_product,
            get_monthly_sales_profit,
        ],
        "system_instruction": """
You are an AI Data Analyst Agent.

You have access to Python analysis tools.

When the user asks a question about the dataset:
1. Decide which tool is required.
2. Call ONLY the necessary tool.
3. Use the tool result to answer.
4. Never invent numbers.
5. Give a concise business-focused answer.

Do not calculate dataset values yourself when an appropriate tool exists.
"""
    }
)


# --------------------------------------------------
# TEST QUESTIONS
# --------------------------------------------------

questions = [
    "Which product has the highest profit?",
    "What is the total quantity sold?",
    "Which region has the highest sales?"
]


for question in questions:

    print("\n" + "=" * 60)
    print("USER QUESTION:")
    print(question)

    response = chat.send_message(question)

    print("\nAI ANALYST:")
    print(response.text)