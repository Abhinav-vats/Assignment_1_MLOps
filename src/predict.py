import requests

url = 'http://127.0.0.1:5005/predict'
headers = {'Content-Type': 'application/json'}
data = {
    "features": [15.78, 17.89, 103.6, 781.0, 0.097, 0.1292, 0.09989, 0.05656, 0.1864, 0.06315, 0.8335, 1.239, 5.158, 94.44, 0.006399, 0.02794, 0.0243, 0.01088, 0.01655, 0.003158, 17.82, 22.1, 113.3, 986.7, 0.1197, 0.2656, 0.243, 0.1085, 0.277, 0.08321]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
