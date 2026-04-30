import requests

BASE_URL = "http://localhost:8000"

# Test health
health = requests.get(f"{BASE_URL}/health")
print(f"Health: {health.json()}")

# Test a question
response = requests.post(
    f"{BASE_URL}/ask",
    json={"question": "What does %DATE do in IBMi RPG?"}
)
print(f"Status: {response.status_code}")
print(f"Answer: {response.json()['answer'][:200]}...")
print(f"Time:   {response.json()['response_time']}s")