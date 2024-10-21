import re
from flask import Flask, request, jsonify
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForNextSentencePrediction
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from openai import OpenAI
import logging
import sys
from sklearn.metrics import silhouette_score, davies_bouldin_score

import concurrent.futures

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
sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device='cuda')
sbert_model.to(device)

# 전역 변수 설정
MAX_THREADS = 15
MAX_TOPIC_LENGTH = 100
DEFAULT_THRESHOLD = 0.7 # 결합된 점수의 임계값
MAX_TOPIC_MESSAGES = 10
TIME_WEIGHT_FACTOR = 0.5
MAX_TIME_WEIGHT = 0.2
MAX_USER_WEIGHT = 0.2
TIME_WINDOW_MINUTES = 4
THREAD_TIMEOUT = timedelta(hours=24)
MEANINGLESS_CHAT_PATTERN = re.compile(r'^([ㄱ-ㅎㅏ-ㅣ]+|[ㅋㅎㄷ]+|[ㅠㅜ]+|[.]+|[~]+|[!]+|[?]+)+$')
CHATS_PER_GROUP = 100
CURRENT_GROUP = 1

# print 관련 설정
DEBUG_MODE = False

# key 전역변수들
MESSAGE_KEY = 'content'
TIME_KEY = 'createdAt'
USER_KEY = 'nickname'
ID_KEY = 'id'
TOPIC_KEY = 'topic'

user_per_threads = dict()

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

def calculate_user_weight(current_user, index):
    global user_per_threads
    
    if (index not in user_per_threads):
        user_per_threads[index] = [current_user]
    
    else:
        # 없으면 새 유저 추가
        user_per_threads[index].append(current_user) 
        
    # 가장 오래된 유저 추방
    if (len(user_per_threads[index]) > MAX_TOPIC_MESSAGES):
        user_per_threads[index].pop(0)

    weight = user_per_threads[index].count(current_user)

    return weight
            

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
                 .sort('_id', 1)
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
        elif (chat[TIME_KEY] - current_combined[TIME_KEY]).total_seconds() <= 60 and chat[USER_KEY] == current_combined[USER_KEY]:
            current_combined[MESSAGE_KEY] += ' ' + chat[MESSAGE_KEY]
            current_combined['original_chats'].append(chat)
        else:
            combined_chats.append(current_combined)
            current_combined = chat.copy()
            current_combined['original_chats'] = [chat]
    
    if current_combined:
        combined_chats.append(current_combined)
    
    return combined_chats

# embedding_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
def get_embeddings(messages):
    embeddings = sbert_model.encode(messages, convert_to_tensor=True).to(device)
    # embedding_text = embedding_client.embeddings.create(input=messages, model='text-embedding-3-small').data
    # result = torch.tensor([message.embedding for message in embedding_text])

    return embeddings
    

def cosine_sim(base_messages, target_message):  
    base_embeddings = sbert_model.encode(base_messages, convert_to_tensor=True).to(device)
    target_embedding = sbert_model.encode([target_message], convert_to_tensor=True).to(device)
    similarities = F.cosine_similarity(base_embeddings, target_embedding, dim=-1)

    return similarities

def combined_score(nsp_model, base_messages, target_message, nsp_weight=0.55, cosine_weight=0.45):
    nsp_results, nsp_probs = nsp_model(base_messages, [target_message] * len(base_messages))
    cosine_sims = cosine_sim(base_messages, target_message)
    
    # combined_scores = nsp_weight * nsp_probs.cpu().numpy() + cosine_weight * cosine_sims
    combined_scores = nsp_weight * nsp_probs + cosine_weight * cosine_sims
    return combined_scores

def assign_topic_with_combined_model(nsp_model, topics, content, current_user, current_time, topic_times, threshold=DEFAULT_THRESHOLD):
    active_topics = []
    active_topic_indices = []

    for idx, (topic, time) in enumerate(zip(topics, topic_times)):
        if current_time - time <= THREAD_TIMEOUT:
            active_topics.append(" ".join(topic[-MAX_TOPIC_MESSAGES:]))
            active_topic_indices.append(idx)

    if not active_topics:
        return len(topics)  # 모든 스레드가 비활성화되었다면 새 스레드 생성

    scores = combined_score(nsp_model, active_topics, content).to(device)
    
    print(f'active topics: {active_topic_indices}, {active_topics}')

    # 시간 가중치 적용 (덧셈)
    time_weights = torch.tensor([calculate_time_weight(current_time, topic_times[i]) for i in active_topic_indices]).to(device)
    user_weights = torch.tensor([calculate_user_weight(current_user, index) for index in active_topic_indices]).to(device)
    user_weights = MAX_USER_WEIGHT * user_weights / sum(user_weights)
    weighted_scores = scores + time_weights + user_weights
    
    max_score = torch.max(weighted_scores)
    
    if max_score > threshold:
        assigned_topic = active_topic_indices[torch.argmax(weighted_scores)]
    else:
        assigned_topic = len(topics)
    
    print(f"Weighted combined scores: {weighted_scores}")
    print(f"Active topics: {[idx + 1 for idx in active_topic_indices]}")
    print(f"Assigned topic: {assigned_topic + 1}")
    
    return assigned_topic

def assign_topics(chats, threshold=DEFAULT_THRESHOLD, max_topics=MAX_THREADS):
    
    if not chats:
        return {}, []

    combined_chats = combine_consecutive_chats(chats)
    
    topics = [[combined_chats[0][MESSAGE_KEY]]]
    topic_mapping = {str(chat[ID_KEY]): 0 for chat in combined_chats[0]['original_chats']}
    topic_times = [combined_chats[0][TIME_KEY]]
    last_speaker_info = {combined_chats[0][USER_KEY]: (combined_chats[0][TIME_KEY], 0)}
    
    nsp_model = NSP(klue_tokenizer, klue_model, device)
    
    for i, combined_chat in enumerate(combined_chats[1:], 1):
        content = combined_chat[MESSAGE_KEY]
        current_time = combined_chat[TIME_KEY]
        current_user = combined_chat[USER_KEY]
        current_speaker = combined_chat[USER_KEY]

        if not is_meaningful_chat(content):
            print(f"Combined Chat {i}: Skipped (meaningless): {content}")
            print('')
            for chat in combined_chat['original_chats']:
                topic_mapping[str(chat[ID_KEY])] = -1
            continue

        print(f"Combined Chat {i}: Content: {content}")
        
        if current_speaker in last_speaker_info:
            last_time, last_topic = last_speaker_info[current_speaker]
            if (current_time - last_time).total_seconds() <= 60:
                assigned_topic = last_topic
                print(f"Combined Chat {i}: Same speaker within 1 minute, assigned to topic {assigned_topic + 1}")
            else:
                assigned_topic = assign_topic_with_combined_model(nsp_model, topics, content, current_user, current_time, topic_times, threshold=threshold)
        else:
            assigned_topic = assign_topic_with_combined_model(nsp_model, topics, content, current_user, current_time, topic_times, threshold=threshold)
        
        if assigned_topic >= len(topics):
            topics.append([content])
            topic_times.append(current_time)
        else:
            topics[assigned_topic].append(content)
            if len(topics[assigned_topic]) > MAX_TOPIC_MESSAGES:
                topics[assigned_topic] = topics[assigned_topic][-MAX_TOPIC_MESSAGES:]
        
        for chat in combined_chat['original_chats']:
            topic_mapping[str(chat[ID_KEY])] = assigned_topic
            print(f"Chat {chats.index(chat) + 1}: Content: {chat[MESSAGE_KEY]}")
            print(f"Chat {chats.index(chat) + 1}: Assigned to topic {assigned_topic + 1}")
        
        topic_times[assigned_topic] = current_time
        last_speaker_info[current_speaker] = (current_time, assigned_topic)
        
        print(f"Thread content for topic {assigned_topic + 1}:")
        print("\n".join(topics[assigned_topic]))
        print('')
    
    return topic_mapping, topics

def model_predict(chats, channel_id, threshold):
    print(f"Room ID: {channel_id}")
    print(f"Combined Threshold: {threshold}")
    print(f"Time Weight Factor: {TIME_WEIGHT_FACTOR}")
    print(f"Max Time Weight: {MAX_TIME_WEIGHT}")
    print(f"Time Window (minutes): {TIME_WINDOW_MINUTES}")
    print(f"Thread Timeout: {THREAD_TIMEOUT}")
    
    topic_mapping, topics = assign_topics(chats=chats, threshold=threshold)
    
    result = []
    
    for chat in chats:
        chat_id = str(chat[ID_KEY])
        predicted_topic = str(topic_mapping.get(chat_id, -1))

        result.append({**chat, TOPIC_KEY: predicted_topic})
    
    return result, topics

def deserialize_chat(raw_chat):
    return {
        ID_KEY: raw_chat[ID_KEY],
        USER_KEY: raw_chat[USER_KEY],
        MESSAGE_KEY: raw_chat[MESSAGE_KEY],
        TIME_KEY: datetime.fromisoformat(raw_chat[TIME_KEY])
    }


def find_best_cluster(embeddings_np, labels_list):
    scores = np.array([silhouette_score(embeddings_np, labels) for labels in labels_list])
    # scores = np.array([davies_bouldin_score(embeddings_np, labels) for labels in labels_list])
    print(scores)

    return np.argmax(scores)

def print_chats(chats):
    for chat in chats:
        print(f'[{chat[TOPIC_KEY]}]:\t {chat[MESSAGE_KEY]}')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        channel_id = data.get('channelId')
        raw_chats = data.get('chats')
        chats = [deserialize_chat(chat) for chat in raw_chats][:CHATS_PER_GROUP]
        embeddings = get_embeddings([chat[MESSAGE_KEY] for chat in chats])
        embeddings_np = embeddings.detach().cpu().numpy()

        thresholds = [0.75]

        new_chats_list = []
        topics_list = []
        
        for threshold in thresholds:
            chat, topic = model_predict(chats, channel_id, threshold)
            new_chats_list.append(chat)
            topics_list.append(topic)

        labels_list = np.array([[int(chat[TOPIC_KEY]) for chat in chats] for chats in new_chats_list])

        best_cluster_index = find_best_cluster(embeddings_np, labels_list)

        best_chats = new_chats_list[best_cluster_index]
        best_topics = topics_list[best_cluster_index]

        print_chats(best_chats)
        
        result = {
            'chats': best_chats,
            'topics': best_topics
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()   
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting improved NSP-based CATD model with KoSBERT...")
    
    try:
        port = int(os.environ.get('PORT', 5000))
        print(f"Server started on port {port}")
        app.run(host='0.0.0.0', port=port)

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()
