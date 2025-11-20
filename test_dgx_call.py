from openai import OpenAI

client = OpenAI(
    base_url="http://115.145.179.241:8000/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello from AI TOP ATOM!"}],
)

print(resp.choices[0].message.content)
print(resp.usage)

