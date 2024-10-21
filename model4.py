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
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForNextSentencePrediction, BertTokenizer
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

# HAN 모델용 토크나이저 (KoBERT 토크나이저 사용)
han_tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

# KoSBERT 모델 초기화
sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
sbert_model.to(device)

# 전역 변수 설정
MAX_THREADS = 15
MAX_TOPIC_LENGTH = 100
COMBINED_THRESHOLD = 0.62  # 결합된 점수의 임계값
MAX_TOPIC_MESSAGES = 10
TIME_WEIGHT_FACTOR = 0.4
MAX_TIME_WEIGHT = 0.15
TIME_WINDOW_MINUTES = 4
THREAD_TIMEOUT = timedelta(hours=1.5)
MEANINGLESS_CHAT_PATTERN = re.compile(r'^([ㄱ-ㅎㅏ-ㅣ]+|[ㅋㅎㄷ]+|[ㅠㅜ]+|[.]+|[~]+|[!]+|[?]+)+$')
CHATS_PER_GROUP = 100
CURRENT_GROUP = 1

# HAN 모델 정의
class HAN(nn.Module):
    def __init__(self, vocab_size, embed_size, word_gru_hidden_size, sent_gru_hidden_size, word_gru_num_layers, sent_gru_num_layers, num_classes):
        super(HAN, self).__init__()
        
        self.word_attention = WordAttention(vocab_size, embed_size, word_gru_hidden_size, word_gru_num_layers)
        self.sent_attention = SentenceAttention(word_gru_hidden_size*2, sent_gru_hidden_size, sent_gru_num_layers)
        self.fc = nn.Linear(sent_gru_hidden_size*2, num_classes)
        
    def forward(self, input_tensor):
        # input_tensor shape: [batch_size, num_messages, message_length]
        batch_size, num_messages, message_length = input_tensor.size()
        
        # Reshape for word attention
        reshaped_input = input_tensor.view(batch_size * num_messages, message_length)
        
        sent_output = self.word_attention(reshaped_input)
        sent_output = sent_output.view(batch_size, num_messages, -1)
        
        doc_output = self.sent_attention(sent_output)
        output = self.fc(doc_output)
        return output

class WordAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, gru_hidden_size, gru_num_layers):
        super(WordAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, gru_hidden_size, num_layers=gru_num_layers, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(gru_hidden_size*2, 1)
        
    def forward(self, text):
        # text shape: [batch_size * sent_num, word_num]
        embedded = self.embedding(text)  # [batch_size * sent_num, word_num, embed_size]
        gru_out, _ = self.gru(embedded)  # [batch_size * sent_num, word_num, gru_hidden_size*2]
        
        attention_weights = F.softmax(self.attention(gru_out).squeeze(-1), dim=1)  # [batch_size * sent_num, word_num]
        sent_vector = torch.bmm(attention_weights.unsqueeze(1), gru_out).squeeze(1)  # [batch_size * sent_num, gru_hidden_size*2]
        
        return sent_vector

class SentenceAttention(nn.Module):
    def __init__(self, sent_gru_input_size, gru_hidden_size, gru_num_layers):
        super(SentenceAttention, self).__init__()
        
        self.gru = nn.GRU(sent_gru_input_size, gru_hidden_size, num_layers=gru_num_layers, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(gru_hidden_size*2, 1)
        
    def forward(self, sent_vectors):
        gru_out, _ = self.gru(sent_vectors)
        
        attention_weights = F.softmax(self.attention(gru_out).squeeze(-1), dim=1)
        doc_vector = torch.bmm(attention_weights.unsqueeze(1), gru_out).squeeze(1)
        
        return doc_vector

# HAN 모델 초기화
vocab_size = len(han_tokenizer.vocab)  # 토크나이저의 어휘 크기 사용
embed_size = 300
word_gru_hidden_size = 100
sent_gru_hidden_size = 100
word_gru_num_layers = 1
sent_gru_num_layers = 1
num_classes = MAX_THREADS + 1  # 새 스레드 생성 옵션 포함

han_model = HAN(vocab_size, embed_size, word_gru_hidden_size, sent_gru_hidden_size, word_gru_num_layers, sent_gru_num_layers, num_classes)
han_model.to(device)

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
                 .sort([('createdAt', 1), ('_id', 1)])  # _id를 두 번째 정렬 기준으로 추가
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

def han_predict(base_messages, target_message):
    # 모든 메시지를 하나의 리스트로 결합
    all_messages = base_messages + [target_message]
    
    # 토큰화 및 인덱싱
    indexed_messages = [han_tokenizer.encode(msg, add_special_tokens=True, max_length=512, truncation=True) for msg in all_messages]
    
    # 패딩
    max_len = max(len(msg) for msg in indexed_messages)
    padded_messages = [msg + [0] * (max_len - len(msg)) for msg in indexed_messages]
    
    # 텐서로 변환
    input_tensor = torch.tensor(padded_messages).unsqueeze(0).to(device)  # [1, num_messages, max_len]
    
    han_model.eval()  # 평가 모드로 설정
    with torch.no_grad():
        output = han_model(input_tensor)
        probs = F.softmax(output, dim=-1)
    
    # 실제 스레드 수에 맞게 확률 조정
    num_threads = len(base_messages)
    adjusted_probs = probs[0, :num_threads+1]  # 새 스레드 옵션 포함
    
    # 확률 정규화
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    
    return adjusted_probs.cpu().numpy()

def ensemble_score(nsp_model, base_messages, target_message, nsp_weight=0.45, cosine_weight=0.45, han_weight=0.1):
    if not base_messages:
        # 기존 주제가 없는 경우, 새 주제 시작을 위한 점수 반환
        return np.array([0.5])  # 중간 확률로 새 주제 시작

    # NSP 모델 실행
    nsp_results, nsp_probs = nsp_model(base_messages, [target_message] * len(base_messages))
    nsp_scores = nsp_probs.cpu().numpy().flatten()
    
    # Cosine 유사도 계산
    cosine_sims = cosine_sim(base_messages, target_message).flatten()
    
    # HAN 모델 실행
    han_probs = han_predict(base_messages, target_message)
    
    # 새 스레드 생성 옵션 추가
    nsp_scores = np.append(nsp_scores, [0.5])  # 중간 확률로 새 스레드 옵션 추가
    cosine_sims = np.append(cosine_sims, [0.5])  # 중간 유사도로 새 스레드 옵션 추가
    
    # 모든 점수의 길이를 맞춤
    min_length = min(len(nsp_scores), len(cosine_sims), len(han_probs))
    nsp_scores = nsp_scores[:min_length]
    cosine_sims = cosine_sims[:min_length]
    han_probs = han_probs[:min_length]
    
    # 가중치를 적용하여 점수 결합
    combined_scores = (nsp_weight * nsp_scores + 
                       cosine_weight * cosine_sims + 
                       han_weight * han_probs)
    
    return combined_scores

def assign_topic_with_ensemble_model(nsp_model, topics, content, current_time, topic_times):
    active_topics = []
    active_topic_indices = []

    for idx, (topic, time) in enumerate(zip(topics, topic_times)):
        if current_time - time <= THREAD_TIMEOUT:
            active_topics.append(" ".join(topic[-MAX_TOPIC_MESSAGES:]))
            active_topic_indices.append(idx)

    if not active_topics:
        print("No active topics. Starting a new topic.")
        return len(topics)  # 새 주제 시작

    scores = ensemble_score(nsp_model, active_topics, content)
    
    # 시간 가중치 적용 (덧셈)
    time_weights = np.array([calculate_time_weight(current_time, topic_times[i]) for i in active_topic_indices])
    weighted_scores = scores[:len(active_topics)] + time_weights
    
    # 새 스레드 생성 점수
    new_thread_score = scores[-1] if len(scores) > len(active_topics) else 0
    
    max_score = np.max(weighted_scores)
    
    if max_score > COMBINED_THRESHOLD and max_score > new_thread_score:
        assigned_topic = active_topic_indices[np.argmax(weighted_scores)]
    else:
        assigned_topic = len(topics)  # 새 스레드 생성
    
    print(f"Weighted ensemble scores: {weighted_scores}")
    print(f"New thread score: {new_thread_score}")
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
                assigned_topic = assign_topic_with_ensemble_model(nsp_model, topics, content, current_time, topic_times)
        else:
            assigned_topic = assign_topic_with_ensemble_model(nsp_model, topics, content, current_time, topic_times)
        
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
        room_id = '670f6b69306e6f8a9dbb9bf4'
        
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
    print("Starting improved NSP-based CATD model with KoSBERT and HAN...")
    
    try:
        # 테스트를 위한 room_id 설정
        room_id = '670f63f099b17496c844bd41'  # 실제 사용할 room_id로 변경하세요
        
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
            print()

        port = int(os.environ.get('PORT', 5000))
        print(f"Server started on port {port}")
        app.run(host='0.0.0.0', port=port)

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()