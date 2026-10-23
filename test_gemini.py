from google import genai
from utils.config import GEMINI_API_KEY


client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="You are an AI Data Analyst. Say hello and briefly explain what you can do."
)

print(response.text)