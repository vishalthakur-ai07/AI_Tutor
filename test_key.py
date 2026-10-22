import os
from openai import OpenAI

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"])
r = client.chat.completions.create(
    model=os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
    messages=[{"role": "user", "content": "What is 12 x 12? Answer in one line."}],
    max_tokens=50,
)

print(r.choices[0].message.content)