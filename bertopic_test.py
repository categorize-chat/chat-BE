
from flask import Flask, request, jsonify
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# MongoDB connection
mongodb_uri = os.getenv('MONGODB_URI')
client = MongoClient(mongodb_uri)
db = client.get_database()
chat_collection = db['chats']

# Initialize BERTopic model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=sentence_model)

def get_chat_messages(room_id):
    chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1))
    return [chat['content'] for chat in chats]

def classify_topics(messages):
    topics, _ = topic_model.fit_transform(messages)
    return topics

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        room_id = data['room_id']
        
        print("Room ID:", room_id)
        
        # Retrieve chat messages
        messages = get_chat_messages(room_id)
        
        if not messages:
            return jsonify({'error': 'No chat messages found for this room'}), 404
        
        # Classify topics
        topics = classify_topics(messages)
        
        # Print topics
        print("Classified topics:")
        for i, (message, topic) in enumerate(zip(messages, topics)):
            print(f"Message {i}: '{message[:50]}...' - Topic: {topic}")
        
        # Prepare result
        result = {
            'topics': topics.tolist(),
            'message_count': len(messages),
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

def assign_topics(room_id):
    messages = get_chat_messages(room_id)
    topics = classify_topics(messages)
    
    print("Assigned topics:")
    for i, (message, topic) in enumerate(zip(messages, topics)):
        print(f"Message {i}: '{message[:50]}...' - Topic: {topic}")
    
    return dict(zip(range(len(messages)), topics))

if __name__ == '__main__':
    print("Starting BERTopic chat classification server...")
    room_id = '66b0fd658aab9f2bd7a41845'  # Replace with your actual room_id
    
    try:
        port = int(os.environ.get('PORT', 5000))
        print(f"Server started on port {port}")
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")