import requests

response = requests.post('http://localhost:5000/predict', json={'room_id': '66b0fd658aab9f2bd7a41841'})
print(response.json())