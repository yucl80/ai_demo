import requests
import json

url = "https://google.serper.dev/search"

payload = json.dumps({
  "q": "apple inc"
})
headers = {
  'X-API-KEY': 'cd123db7d54b206a3a109e205ae231d90f103902',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
