import json, os, requests
url = os.environ["AML_ENDPOINT_URL"]
key = os.environ["AML_ENDPOINT_KEY"]
req_path = os.path.join(os.path.dirname(__file__), "sample-request.json")
with open(req_path) as f:
    payload = json.load(f)
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
r = requests.post(url, headers=headers, json=payload, timeout=30)
print("HTTP status:", r.status_code)
print("Prediction:", r.text)
