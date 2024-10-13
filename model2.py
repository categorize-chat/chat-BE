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

# BERT 모델 및 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModelForNextSentencePrediction.from_pretrained("klue/bert-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 전역 변수 설정
MAX_THREADS = 15
MAX_TOPIC_LENGTH = 100  # 토픽 내용의 최대 단어 수
NSP_THRESHOLD = 0.85  # 임계값 상향 조정
MAX_TOPIC_MESSAGES = 5  # 각 토픽에서 유지할 최대 메시지 수
TIME_WEIGHT_FACTOR = 0.2  # 시간 가중치 팩터
MAX_TIME_WEIGHT = 0.1  # 최대 시간 가중치
TIME_WINDOW_MINUTES = 5  # 최대 가중치를 갖는 시간 범위 (분)

# 무의미한 채팅을 필터링하기 위한 정규 표현식 패턴
MEANINGLESS_CHAT_PATTERN = re.compile(r'^([ㄱ-ㅎㅏ-ㅣ]+|[ㅋㅎㄷ]+|[ㅠㅜ]+|[.]+|[~]+|[!]+|[?]+)+$')

# 채팅 그룹 전역 변수 설정
CHATS_PER_GROUP = 100
CURRENT_GROUP = 4

class NSP:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def __call__(self, base_messages, target_messages):
        encoding = self.tokenizer(base_messages, target_messages, return_tensors='pt', padding=True, truncation=True)
        
        input_ids = encoding['input_ids'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids)
            logits = outputs.logits
        
        probs = F.softmax(logits, dim=-1)
        
        is_same_class = torch.argmax(probs, dim=-1) == 0
        prob = torch.max(probs, dim=-1).values

        return is_same_class, prob  # PyTorch 텐서를 직접 반환

def calculate_time_weight(current_time, thread_time):
    time_diff = (current_time - thread_time).total_seconds() / 60  # 분 단위로 변환
    if time_diff <= TIME_WINDOW_MINUTES:
        return MAX_TIME_WEIGHT
    else:
        return max(0, MAX_TIME_WEIGHT * np.exp(-TIME_WEIGHT_FACTOR * (time_diff - TIME_WINDOW_MINUTES)))

def is_meaningful_chat(content):
    # 내용이 없거나 공백만 있는 경우
    if not content.strip():
        return False
    
    # 정규 표현식 패턴에 매치되는 경우 (무의미한 채팅)
    if MEANINGLESS_CHAT_PATTERN.match(content.strip()):
        return False
    
    # 한글 2글자 이하 채팅 (한글만 있는 경우)
    if re.match(r'^[가-힣]{1,2}$', content.strip()):
        return False
    
    # 전체 길이가 2자 이하인 경우
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

def assign_topics(room_id, max_topics=MAX_THREADS):
    chats, start_index, end_index = get_chat_group(room_id)
    
    if not chats:
        return {}, []
    
    topics = [[chats[0]['content']]]
    topic_mapping = {str(chats[0]['_id']): 0}
    topic_times = [chats[0]['createdAt']]
    last_speaker_info = {chats[0]['nickname']: (chats[0]['createdAt'], 0)}
    
    nsp_model = NSP(tokenizer, model, device)
    
    for i, chat in enumerate(chats[1:], 1):
        chat_id = str(chat['_id'])
        content = chat['content']
        current_time = chat['createdAt']
        current_speaker = chat['nickname']

        if not is_meaningful_chat(content):
            print(f"Chat {start_index + i + 1}: Skipped (meaningless): {content}")
            print()
            topic_mapping[chat_id] = -1
            continue

        print(f"Chat {start_index + i + 1}: Content: {content}")
        
        if current_speaker in last_speaker_info:
            last_time, last_topic = last_speaker_info[current_speaker]
            if (current_time - last_time).total_seconds() <= 60:
                assigned_topic = last_topic
                print(f"Chat {start_index + i + 1}: Same speaker within 1 minute, assigned to topic {assigned_topic + 1}")
            else:
                assigned_topic = assign_topic_with_nsp(nsp_model, topics, content, current_time, topic_times)
        else:
            assigned_topic = assign_topic_with_nsp(nsp_model, topics, content, current_time, topic_times)
        
        if assigned_topic >= len(topics):
            topics.append([content])
            topic_times.append(current_time)
        else:
            topics[assigned_topic].append(content)
            if len(topics[assigned_topic]) > MAX_TOPIC_MESSAGES:
                topics[assigned_topic] = topics[assigned_topic][-MAX_TOPIC_MESSAGES:]
            topic_times[assigned_topic] = current_time
        
        topic_mapping[chat_id] = assigned_topic
        last_speaker_info[current_speaker] = (current_time, assigned_topic)
        
        print(f"Chat {start_index + i + 1}: Assigned to topic {assigned_topic + 1}")
        print()
    
    return topic_mapping, chats

def assign_topic_with_nsp(nsp_model, topics, content, current_time, topic_times):
    thread_contents = [" ".join(topic[-MAX_TOPIC_MESSAGES:]) for topic in topics]
    target_messages = [content] * len(topics)
    
    _, probs = nsp_model(thread_contents, target_messages)
    
    # 시간 가중치 적용 (덧셈)
    time_weights = torch.tensor([calculate_time_weight(current_time, t) for t in topic_times], device=device)
    weighted_probs = probs + time_weights
    
    max_prob = torch.max(weighted_probs).item()
    assigned_topic = torch.argmax(weighted_probs).item() if max_prob > NSP_THRESHOLD else len(topics)
    
    print(f"Weighted IsNext probabilities: {weighted_probs}")
    
    return assigned_topic

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        room_id = '66b0fd658aab9f2bd7a41841'
        
        print(f"Room ID: {room_id}")
        print(f"NSP Threshold: {NSP_THRESHOLD}")
        print(f"Time Weight Factor: {TIME_WEIGHT_FACTOR}")
        print(f"Max Time Weight: {MAX_TIME_WEIGHT}")
        print(f"Time Window (minutes): {TIME_WINDOW_MINUTES}")
        
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
                'predicted_topic': int(topic_mapping.get(chat_id, -1))  # int64를 int로 변환
            })
        
        print("Result before jsonify:", result)
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()   
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting improved NSP-based CATD model...")
    
    try:
        port = int(os.environ.get('PORT', 5000))
        print(f"Server started on port {port}")
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")