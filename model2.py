from flask import Flask, request, jsonify
import json
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoModel, DistilBertForMaskedLM, TFBertForNextSentencePrediction

# .env 파일 로드
load_dotenv()

app = Flask(__name__)

# MongoDB 연결
mongodb_uri = os.getenv('MONGODB_URI')
if not mongodb_uri or not mongodb_uri.startswith(('mongodb://', 'mongodb+srv://')):
    raise ValueError("Invalid MONGODB_URI. It must start with 'mongodb://' or 'mongodb+srv://'")

client = MongoClient(mongodb_uri)
db = client.get_database()
chat_collection = db['chats']

# BERT 모델 및 토크나이저 초기화
model = TFBertForNextSentencePrediction.from_pretrained('klue/bert-base', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 전역 변수 설정
MAX_THREADS = 15
MAX_TOPIC_LENGTH = 100  # 토픽 내용의 최대 단어 수
NSP_THRESHOLD = 0.85  # 임계값 상향 조정
MAX_TOPIC_MESSAGES = 5  # 각 토픽에서 유지할 최대 메시지 수

def NSP(base_message, target_message):
    encoding = tokenizer(base_message, target_message, return_tensors='tf') 

    logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]

    softmax = tf.keras.layers.Softmax()
    probs = softmax(logits)

    is_same_class = not bool(tf.math.argmax(probs, axis=-1).numpy())
    prob = np.max(probs, axis=-1)[0]

    return is_same_class, prob

def assign_topics(room_id, max_topics=MAX_THREADS):
    chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1))
    topics = []
    topic_mapping = {}
    
    # 첫 번째 채팅은 항상 첫 번째 토픽에 할당
    first_chat = chats[0]
    topic_mapping[str(first_chat['_id'])] = 1
    topics.append([first_chat['content']])
    
    for i, chat in enumerate(chats[1:], 1):
        chat_id = str(chat['_id'])
        content = chat['content']
        assigned_topic = None
        max_prob = 0
        
        for j, topic in enumerate(topics):
            # 각 토픽의 마지막 MAX_TOPIC_MESSAGES 개의 메시지만 사용
            topic_content = " ".join(topic[-MAX_TOPIC_MESSAGES:])
            is_same_class, prob = NSP(topic_content, content)
            print(f"Chat {i+1}, Topic {j+1}: IsNext probability: {prob:.4f}")
            if prob > max_prob:
                max_prob = prob
                if prob > NSP_THRESHOLD:
                    assigned_topic = j + 1
        
        if assigned_topic is None:
            if len(topics) < max_topics:
                topics.append([content])
                assigned_topic = len(topics)
                print(f"Chat {i+1}: New topic created. Topic {assigned_topic}")
            else:
                # 가장 유사한 토픽에 할당
                assigned_topic = max(range(len(topics)), key=lambda x: NSP(" ".join(topics[x][-MAX_TOPIC_MESSAGES:]), content)[1]) + 1
                print(f"Chat {i+1}: Max topics reached. Assigned to most similar topic {assigned_topic}")
        else:
            print(f"Chat {i+1}: Assigned to topic {assigned_topic}")
        
        # 토픽에 새 메시지 추가하고 최대 개수 제한
        topics[assigned_topic - 1].append(content)
        if len(topics[assigned_topic - 1]) > MAX_TOPIC_MESSAGES:
            topics[assigned_topic - 1] = topics[assigned_topic - 1][-MAX_TOPIC_MESSAGES:]
        
        topic_mapping[chat_id] = assigned_topic
    
    return topic_mapping

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        room_id = '66b0fd658aab9f2bd7a41841'
        
        print(f"Room ID: {room_id}")
        print(f"NSP Threshold: {NSP_THRESHOLD}")
        
        chat_count = chat_collection.count_documents({'room': ObjectId(room_id)})
        print(f"Number of chat messages for room {room_id}: {chat_count}")
        
        if chat_count == 0:
            return jsonify({'error': 'No chat messages found for this room'}), 404
        
        topic_mapping = assign_topics(room_id)
        
        chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1))
        result = []
        
        for chat in chats:
            chat_id = str(chat['_id'])
            result.append({
                'content': chat['content'],
                'predicted_topic': topic_mapping.get(chat_id, -1)  # -1 if not found
            })
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()   
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting NSP-based CATD model...")
    
    try:
        port = int(os.environ.get('PORT', 5000))
        print(f"Server started on port {port}")
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")