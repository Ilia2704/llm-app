from openai import OpenAI
import os

YC_API_KEY = os.environ["YC_API_KEY"]
YC_FOLDER_ID = os.environ["YC_FOLDER_ID"]

client = OpenAI(
    base_url="https://llm.api.cloud.yandex.net/v1",
    api_key="DUMMY",  # не используется, если переопределим заголовок
    default_headers={
        "Authorization": f"Api-Key {YC_API_KEY}",
        "OpenAI-Project": YC_FOLDER_ID,   # важно для совместимости
    },
)

print(client.models.list())
