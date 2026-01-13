import requests

url = "http://127.0.0.1:5000/predict"

# Replace these 24 values with your normalized energy values
data = {
    "last_24_values": [0.1, 0.15, 0.2, 0.18, 0.12, 0.1, 0.05, 0.07, 0.12, 0.18,
                       0.22, 0.25, 0.2, 0.18, 0.16, 0.15, 0.14, 0.1, 0.08, 0.05,
                       0.03, 0.02, 0.01, 0.0]
}

response = requests.post(url, json=data)
print(response.json())
