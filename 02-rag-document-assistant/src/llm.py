import os

import dotenv
from groq import Groq

# Load environment variables from .env
dotenv.load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing! Check your .env file.")

# Initialize Groq client
client = Groq(api_key=api_key)


def generate_answer(prompt: str) -> str:
    """
    Generate an answer using Groq LLM.

    Args:
        prompt (str): User input prompt.

    Returns:
        str: Generated response.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_completion_tokens=1024
    )

    return response.choices[0].message.content