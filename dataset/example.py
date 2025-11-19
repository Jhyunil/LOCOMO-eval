from openai import OpenAI
import os

client = OpenAI()                     # 환경변수 OPENAI_API_KEY 자동 인식
# 또는 client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",              # 원하는 모델
    messages=[{"role": "user", "content": "안녕, OpenAI!"}],
    temperature=0.7,
)

print(response.choices[0].message.content)

