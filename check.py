from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env
load_dotenv()

client = OpenAI()  # API key is taken from OPENAI_API_KEY

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Say OK if the API key works."}
    ],
    temperature=0
)

print(response.choices[0].message.content)
