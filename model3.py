import re
from flask import Flask, request, jsonify
import json
import tensorflow as tf
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForNextSentencePrediction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

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

# KLUE BERT 모델 및 토크나이저 초기화
klue_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
klue_model = AutoModelForNextSentencePrediction.from_pretrained("klue/bert-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
klue_model.to(device)

# KoSBERT 모델 초기화
sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
sbert_model.to(device)

# 전역 변수 설정
MAX_THREADS = 15
MAX_TOPIC_LENGTH = 100
COMBINED_THRESHOLD = 0.7 # 결합된 점수의 임계값
MAX_TOPIC_MESSAGES = 10
TIME_WEIGHT_FACTOR = 0.5
MAX_TIME_WEIGHT = 0.2
TIME_WINDOW_MINUTES = 4
THREAD_TIMEOUT = timedelta(hours=1)
MEANINGLESS_CHAT_PATTERN = re.compile(r'^([ㄱ-ㅎㅏ-ㅣ]+|[ㅋㅎㄷ]+|[ㅠㅜ]+|[.]+|[~]+|[!]+|[?]+)+$')
CHATS_PER_GROUP = 100
CURRENT_GROUP = 1

class NSP:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def __call__(self, base_messages, target_messages):
        encoding = self.tokenizer(base_messages, target_messages, return_tensors='pt', padding=True, truncation=True)
        
        input_ids = encoding['input_ids'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)  
        
        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask) 
            logits = outputs.logits
        
        probs = F.softmax(logits, dim=-1)
        
        is_same_class = torch.argmax(probs, dim=-1) == 0
        prob = torch.max(probs, dim=-1).values

        return is_same_class, prob

def calculate_time_weight(current_time, thread_time):
    time_diff = (current_time - thread_time).total_seconds() / 60
    if time_diff <= TIME_WINDOW_MINUTES:
        return MAX_TIME_WEIGHT
    else:
        return max(0, MAX_TIME_WEIGHT * np.exp(-TIME_WEIGHT_FACTOR * (time_diff - TIME_WINDOW_MINUTES)))

def is_meaningful_chat(content):
    if not content.strip():
        return False
    if MEANINGLESS_CHAT_PATTERN.match(content.strip()):
        return False
    if re.match(r'^[가-힣]{1,2}$', content.strip()):
        return False
    if len(content.strip()) <= 2:
        return False
    return True

def get_chat_group(room_id):
    total_chats = chat_collection.count_documents({'room': ObjectId(room_id)})
    start_index = (CURRENT_GROUP - 1) * CHATS_PER_GROUP
    end_index = min(start_index + CHATS_PER_GROUP, total_chats)
    
    chats = list(chat_collection.find({'room': ObjectId(room_id)})
                 .sort('createdAt', 1)
                 .skip(start_index)
                 .limit(CHATS_PER_GROUP))
    
    return chats, start_index, end_index

def combine_consecutive_chats(chats):
    combined_chats = []
    current_combined = None
    
    for chat in chats:
        if current_combined is None:
            current_combined = chat.copy()
            current_combined['original_chats'] = [chat]
        elif (chat['createdAt'] - current_combined['createdAt']).total_seconds() <= 60 and chat['nickname'] == current_combined['nickname']:
            current_combined['content'] += ' ' + chat['content']
            current_combined['original_chats'].append(chat)
        else:
            combined_chats.append(current_combined)
            current_combined = chat.copy()
            current_combined['original_chats'] = [chat]
    
    if current_combined:
        combined_chats.append(current_combined)
    
    return combined_chats

def cosine_sim(base_messages, target_message):
    base_embeddings = sbert_model.encode(base_messages)
    target_embedding = sbert_model.encode([target_message])
    similarities = cosine_similarity(base_embeddings, target_embedding)
    return similarities.flatten()

def combined_score(nsp_model, base_messages, target_message, nsp_weight=0.55, cosine_weight=0.45):
    nsp_results, nsp_probs = nsp_model(base_messages, [target_message] * len(base_messages))
    cosine_sims = cosine_sim(base_messages, target_message)
    
    combined_scores = nsp_weight * nsp_probs.cpu().numpy() + cosine_weight * cosine_sims
    return combined_scores

def assign_topic_with_combined_model(nsp_model, topics, content, current_time, topic_times):
    active_topics = []
    active_topic_indices = []

    for idx, (topic, time) in enumerate(zip(topics, topic_times)):
        if current_time - time <= THREAD_TIMEOUT:
            active_topics.append(" ".join(topic[-MAX_TOPIC_MESSAGES:]))
            active_topic_indices.append(idx)

    if not active_topics:
        return len(topics)  # 모든 스레드가 비활성화되었다면 새 스레드 생성

    scores = combined_score(nsp_model, active_topics, content)
    
    # 시간 가중치 적용 (덧셈)
    time_weights = np.array([calculate_time_weight(current_time, topic_times[i]) for i in active_topic_indices])
    weighted_scores = scores + time_weights
    
    max_score = np.max(weighted_scores)
    
    if max_score > COMBINED_THRESHOLD:
        assigned_topic = active_topic_indices[np.argmax(weighted_scores)]
    else:
        assigned_topic = len(topics)
    
    print(f"Weighted combined scores: {weighted_scores}")
    print(f"Active topics: {[idx + 1 for idx in active_topic_indices]}")
    print(f"Assigned topic: {assigned_topic + 1}")
    
    return assigned_topic

def assign_topics(room_id, max_topics=MAX_THREADS):
    chats, start_index, end_index = get_chat_group(room_id)
    
    if not chats:
        return {}, [], []

    combined_chats = combine_consecutive_chats(chats)
    
    topics = [[combined_chats[0]['content']]]
    topic_mapping = {str(chat['_id']): 0 for chat in combined_chats[0]['original_chats']}
    topic_times = [combined_chats[0]['createdAt']]
    last_speaker_info = {combined_chats[0]['nickname']: (combined_chats[0]['createdAt'], 0)}
    
    nsp_model = NSP(klue_tokenizer, klue_model, device)
    
    for i, combined_chat in enumerate(combined_chats[1:], 1):
        content = combined_chat['content']
        current_time = combined_chat['createdAt']
        current_speaker = combined_chat['nickname']

        if not is_meaningful_chat(content):
            print(f"Combined Chat {i}: Skipped (meaningless): {content}")
            print()
            for chat in combined_chat['original_chats']:
                topic_mapping[str(chat['_id'])] = -1
            continue

        print(f"Combined Chat {i}: Content: {content}")
        
        if current_speaker in last_speaker_info:
            last_time, last_topic = last_speaker_info[current_speaker]
            if (current_time - last_time).total_seconds() <= 60:
                assigned_topic = last_topic
                print(f"Combined Chat {i}: Same speaker within 1 minute, assigned to topic {assigned_topic + 1}")
            else:
                assigned_topic = assign_topic_with_combined_model(nsp_model, topics, content, current_time, topic_times)
        else:
            assigned_topic = assign_topic_with_combined_model(nsp_model, topics, content, current_time, topic_times)
        
        if assigned_topic >= len(topics):
            topics.append([content])
            topic_times.append(current_time)
        else:
            topics[assigned_topic].append(content)
            if len(topics[assigned_topic]) > MAX_TOPIC_MESSAGES:
                topics[assigned_topic] = topics[assigned_topic][-MAX_TOPIC_MESSAGES:]
        
        for chat in combined_chat['original_chats']:
            topic_mapping[str(chat['_id'])] = assigned_topic
            print(f"Chat {start_index + chats.index(chat) + 1}: Content: {chat['content']}")
            print(f"Chat {start_index + chats.index(chat) + 1}: Assigned to topic {assigned_topic + 1}")
        
        topic_times[assigned_topic] = current_time
        last_speaker_info[current_speaker] = (current_time, assigned_topic)
        
        print(f"Thread content for topic {assigned_topic + 1}:")
        print("\n".join(topics[assigned_topic]))
        print()
    
    return topic_mapping, chats, topics

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        room_id = '66b0fd658aab9f2bd7a41842'
        
        print(f"Room ID: {room_id}")
        print(f"Combined Threshold: {COMBINED_THRESHOLD}")
        print(f"Time Weight Factor: {TIME_WEIGHT_FACTOR}")
        print(f"Max Time Weight: {MAX_TIME_WEIGHT}")
        print(f"Time Window (minutes): {TIME_WINDOW_MINUTES}")
        print(f"Thread Timeout: {THREAD_TIMEOUT}")
        
        chat_count = chat_collection.count_documents({'room': ObjectId(room_id)})
        print(f"Number of chat messages for room {room_id}: {chat_count}")
        
        if chat_count == 0:
            return jsonify({'error': 'No chat messages found for this room'}), 404
        
        topic_mapping, chats, topics = assign_topics(room_id)
        
        result = []
        
        for chat in chats:
            chat_id = str(chat['_id'])
            predicted_topic = int(topic_mapping.get(chat_id, -1))
            result.append({
                'content': chat['content'],
                'predicted_topic': predicted_topic,
                'thread_content': topics[predicted_topic] if predicted_topic >= 0 and predicted_topic < len(topics) else []
            })
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()   
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting improved NSP-based CATD model with KoSBERT...")
    
    try:
        # 테스트를 위한 room_id 설정
        room_id = '66b0fd658aab9f2bd7a41842'  # 실제 사용할 room_id로 변경하세요
        
        print(f"Room ID: {room_id}")
        print(f"Combined Threshold: {COMBINED_THRESHOLD}")
        print(f"Time Weight Factor: {TIME_WEIGHT_FACTOR}")
        print(f"Max Time Weight: {MAX_TIME_WEIGHT}")
        print(f"Time Window (minutes): {TIME_WINDOW_MINUTES}")
        print(f"Thread Timeout: {THREAD_TIMEOUT}")
        
        chat_count = chat_collection.count_documents({'room': ObjectId(room_id)})
        print(f"Number of chat messages for room {room_id}: {chat_count}")
        
        if chat_count == 0:
            print('No chat messages found for this room')
        else:
            topic_mapping, chats, topics = assign_topics(room_id)
            
            print("\nResults:")
            for chat in chats:
                chat_id = str(chat['_id'])
                predicted_topic = int(topic_mapping.get(chat_id, -1))
                print(f"Chat: {chat['content']}")
                print(f"Predicted Topic: {predicted_topic}")
                if predicted_topic >= 0 and predicted_topic < len(topics):
                    print(f"Thread Content: {topics[predicted_topic]}")
                print()

        port = int(os.environ.get('PORT', 5000))
        print(f"Server started on port {port}")
        app.run(host='0.0.0.0', port=port)

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()