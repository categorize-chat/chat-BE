import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import requests
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
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

def get_model_prediction(room_id):
    try:
        response = requests.post(MODEL_SERVER_URL, 
                                 json={'room_id': room_id},
                                 headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        response_data = json.loads(response.text)
        
        print("Model server response:")
        for item in response_data:
            print(f"Content: {item['content']}, Predicted Topic: {item['predicted_topic']}")
        
        return [item['predicted_topic'] for item in response_data]
    except requests.RequestException as e:
        print(f"Error calling model server: {e}")
        if hasattr(e.response, 'text'):
            print(f"Server response: {e.response.text}")
        return None

def compare_topics(room_id):
    chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1))
    
    if not chats:
        print(f"No chats found for room_id: {room_id}")
        return
    
    actual_topics = [int(chat.get('topic', '-1')) for chat in chats[1:]]  # Skip the first message and use -1 as default for unknown topics
    predicted_topics = get_model_prediction(room_id)
    
    if predicted_topics is None:
        print("Failed to get predictions from the model server.")
        return
    
    if len(predicted_topics) != len(actual_topics):
        print(f"Mismatch in the number of topics. Actual: {len(actual_topics)}, Predicted: {len(predicted_topics)}")
        # Truncate to the shorter length
        min_length = min(len(actual_topics), len(predicted_topics))
        actual_topics = actual_topics[:min_length]
        predicted_topics = predicted_topics[:min_length]
    
    # Detailed analysis
    print("\nDetailed Analysis:")
    print("Actual topic distribution:")
    print(Counter(actual_topics))
    print("\nPredicted topic distribution:")
    print(Counter(predicted_topics))
    
    if not actual_topics or not predicted_topics:
        print("No topics to compare. Exiting analysis.")
        return
    
    # Calculate and print metrics
    try:
        accuracy = accuracy_score(actual_topics, predicted_topics)
        conf_matrix = confusion_matrix(actual_topics, predicted_topics)
        class_report = classification_report(actual_topics, predicted_topics, zero_division=0)
        
        print("\nOverall Results:")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
        
        # Analyze misclassifications
        misclassifications = [(act, pred) for act, pred in zip(actual_topics, predicted_topics) if act != pred]
        if misclassifications:
            print("\nMisclassification Analysis:")
            misclassification_counts = Counter(misclassifications)
            for (act, pred), count in misclassification_counts.most_common():
                print(f"Actual: {act}, Predicted: {pred}, Count: {count}")
        else:
            print("\nNo misclassifications found.")
    except Exception as e:
        print(f"Error during metric calculation: {str(e)}")

    # Print detailed comparison
    print("\nDetailed Comparison:")
    for i, (actual, predicted) in enumerate(zip(actual_topics, predicted_topics)):
        print(f"Message {i+1}: Actual Topic: {actual}, Predicted Topic: {predicted}")
        if i < len(chats) - 1:  # Skip the first chat as we did for topics
            print(f"Content: {chats[i+1]['content']}")
        print("---")

if __name__ == '__main__':
    room_id = '66b0fd658aab9f2bd7a41841'  # Replace with your actual room ID
    compare_topics(room_id)