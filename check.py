from openai import OpenAI

client = OpenAI()

response = client.responses.create(
  model="gpt-5-nano",
  input="write a haiku about ai",
  store=True,
)

print(response);
