import requests

url = "http://127.0.0.1:8000/extract-text/"
files = {'file': open('doc.png', 'rb')}  # Replace with the path to your image

response = requests.post(url, files=files)

if response.status_code == 200:
    lines = response.json().get('lines', [])
    for line in lines:
        print(line)
else:
    print("Error:", response.status_code)
