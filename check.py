from openai import OpenAI

client = OpenAI()

response = client.responses.create(
  model="gpt-4-nano",
  input="write a haiku about ai",
  store=True,
)

print('response:', response);