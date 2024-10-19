import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import requests
import json

# Load environment variables
load_dotenv()

# MongoDB connection
mongodb_uri = os.getenv('MONGODB_URI')
client = MongoClient(mongodb_uri)
db = client.get_database()
chat_collection = db['chats']

# Connect to the model server
MODEL_SERVER_URL = 'http://localhost:5000/predict'

def json_serialize_object_id(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def get_model_prediction(room_id):
    try:
        response = requests.post(MODEL_SERVER_URL, 
                                 json={'room_id': room_id},
                                 headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        response_data = response.json()
        
        print("Model server response:", json.dumps(response_data, indent=2))
        
        return response_data
    except requests.RequestException as e:
        print(f"Error calling model server: {e}")
        if hasattr(e.response, 'text'):
            print(f"Server response: {e.response.text}")
        return None

def predict_topics(room_id, limit=200):
    chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1).limit(limit))
    
    if not chats:
        print(f"No chats found for room_id: {room_id}")
        return
    
    predictions = get_model_prediction(room_id)
    
    if predictions is None:
        print("Failed to get predictions from the model server.")
        return
    
    print("\nPrediction Results:")
    for prediction in predictions:
        print(f"Content: {prediction['content']}")
        print(f"Predicted Topic: {prediction['predicted_topic']}")
        print(f"Thread Content: {prediction['thread_content']}")
        print("---")

if __name__ == '__main__':
    room_id = '66b0fd658aab9f2bd7a41842'  # Replace with your actual room ID
    predict_topics(room_id)