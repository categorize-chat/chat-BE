import re
from quart import Quart, request, jsonify
import json
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForNextSentencePrediction, BertTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import numpy as np
import os
from dotenv import load_dotenv
import asyncio
from openai import OpenAI
import traceback
from copy import deepcopy

# .env 파일 로드
load_dotenv()

app = Quart(__name__)

# 전역 변수 설정
PARAMETER_SETS = {
    'low': {
        'combined_threshold': 0.6,
        'consecutive_time': 60
    },
    'mid': {
        'combined_threshold': 0.65,
        'consecutive_time': -1
    },
    'high': {
        'combined_threshold': 0.7,
        'consecutive_time': -1
    }
}
MAX_THREADS = 30
MAX_TOPIC_LENGTH = 100
MAX_TOPIC_MESSAGES = 10
TIME_WEIGHT_FACTOR = 0.4
MAX_TIME_WEIGHT = 0.15
TIME_WINDOW_MINUTES = 4
THREAD_TIMEOUT = timedelta(hours=12)
MEANINGLESS_CHAT_PATTERN = re.compile(r'^([ㄱ-ㅎㅏ-ㅣ]+|[ㅋㅎㄷ]+|[ㅠㅜ]+|[.]+|[~]+|[!]+|[?]+)+$')

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

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# HAN 모델 정의
class HAN(nn.Module):
    def __init__(self, vocab_size, embed_size, word_gru_hidden_size, sent_gru_hidden_size, word_gru_num_layers, sent_gru_num_layers, num_classes):
        super(HAN, self).__init__()
        
        self.word_attention = WordAttention(vocab_size, embed_size, word_gru_hidden_size, word_gru_num_layers)
        self.sent_attention = SentenceAttention(word_gru_hidden_size*2, sent_gru_hidden_size, sent_gru_num_layers)
        self.fc = nn.Linear(sent_gru_hidden_size*2, num_classes)
        
    def forward(self, input_tensor):
        batch_size, num_messages, message_length = input_tensor.size()
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
        embedded = self.embedding(text)
        gru_out, _ = self.gru(embedded)
        attention_weights = F.softmax(self.attention(gru_out).squeeze(-1), dim=1)
        sent_vector = torch.bmm(attention_weights.unsqueeze(1), gru_out).squeeze(1)
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
vocab_size = len(han_tokenizer.vocab)
embed_size = 300
word_gru_hidden_size = 100
sent_gru_hidden_size = 100
word_gru_num_layers = 1
sent_gru_num_layers = 1
num_classes = MAX_THREADS + 1

han_model = HAN(vocab_size, embed_size, word_gru_hidden_size, sent_gru_hidden_size, 
                word_gru_num_layers, sent_gru_num_layers, num_classes)
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

def combine_consecutive_chats(chats, consecutive_time):
    if consecutive_time < 0:
        return [{'id': chat['id'], 'content': chat['content'], 
                'createdAt': chat['createdAt'], 'nickname': chat['nickname'],
                'original_chats': [chat]} for chat in chats]
    
    combined_chats = []
    current_combined = None
    
    for chat in chats:
        if current_combined is None:
            current_combined = chat.copy()
            current_combined['original_chats'] = [chat]
        elif (chat['createdAt'] - current_combined['createdAt']).total_seconds() <= consecutive_time and chat['nickname'] == current_combined['nickname']:
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
    all_messages = base_messages + [target_message]
    indexed_messages = [han_tokenizer.encode(msg, add_special_tokens=True, max_length=512, truncation=True) 
                       for msg in all_messages]
    max_len = max(len(msg) for msg in indexed_messages)
    padded_messages = [msg + [0] * (max_len - len(msg)) for msg in indexed_messages]
    input_tensor = torch.tensor(padded_messages).unsqueeze(0).to(device)
    
    han_model.eval()
    with torch.no_grad():
        output = han_model(input_tensor)
        probs = F.softmax(output, dim=-1)
    
    num_threads = len(base_messages)
    adjusted_probs = probs[0, :num_threads+1]
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    
    return adjusted_probs.cpu().numpy()

def ensemble_score(nsp_model, base_messages, target_message, nsp_weight=0.45, cosine_weight=0.45, han_weight=0.1):
    if not base_messages:
        return np.array([0.5])

    nsp_results, nsp_probs = nsp_model(base_messages, [target_message] * len(base_messages))
    nsp_scores = nsp_probs.cpu().numpy().flatten()
    
    cosine_sims = cosine_sim(base_messages, target_message).flatten()
    
    han_probs = han_predict(base_messages, target_message)
    
    nsp_scores = np.append(nsp_scores, [0.5])
    cosine_sims = np.append(cosine_sims, [0.5])
    
    min_length = min(len(nsp_scores), len(cosine_sims), len(han_probs))
    nsp_scores = nsp_scores[:min_length]
    cosine_sims = cosine_sims[:min_length]
    han_probs = han_probs[:min_length]
    
    combined_scores = (nsp_weight * nsp_scores + 
                      cosine_weight * cosine_sims + 
                      han_weight * han_probs)
    
    return combined_scores

def assign_topic_with_ensemble_model(nsp_model, topics, content, current_time, topic_times, combined_threshold):
    active_topics = []
    active_topic_indices = []

    for idx, (topic, time) in enumerate(zip(topics, topic_times)):
        if current_time - time <= THREAD_TIMEOUT:
            active_topics.append(" ".join(topic[-MAX_TOPIC_MESSAGES:]))
            active_topic_indices.append(idx)

    if not active_topics:
        print("No active topics. Starting a new topic.")
        return len(topics)

    scores = ensemble_score(nsp_model, active_topics, content)
    
    time_weights = np.array([calculate_time_weight(current_time, topic_times[i]) 
                            for i in active_topic_indices])
    weighted_scores = scores[:len(active_topics)] + time_weights
    
    new_thread_score = scores[-1] if len(scores) > len(active_topics) else 0
    
    max_score = np.max(weighted_scores)
    
    if max_score > combined_threshold and max_score > new_thread_score:
        assigned_topic = active_topic_indices[np.argmax(weighted_scores)]
    else:
        assigned_topic = len(topics)
    
    return assigned_topic

async def summarize_all_topics(topics_data):
    try:
        formatted_topics = []
        for topic_id, messages in topics_data.items():
            topic_content = "\n".join(messages)
            formatted_topics.append(f"[Topic {topic_id}]\n{topic_content}")
        
        all_content = "\n\n".join(formatted_topics)
        
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """여러 개의 대화 토픽이 주어집니다. 각 토픽은 [Topic N] 형식으로 구분되어 있습니다.
                각 토픽별로 다음 정보를 제공해주세요:
                1. 해당 토픽의 가장 중요한 키워드 1개 - 키워드는 다른 토픽들과 겹치면 안됩니다.
                2. 대화 내용의 간단한 요약
                
                다음과 같은 JSON 형식으로 응답해주세요:
                {
                    "N": {"keywords": ["키워드"], "content": "요약"},
                    ...
                }"""},
                {"role": "user", "content": all_content}
            ],
            response_format={ "type": "json_object" }
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error in summarizing topics: {str(e)}")
        return {}

async def assign_topics_with_params(chats, combined_threshold, consecutive_time):
    if not chats:
        return {}, [], [], {}

    combined_chats = combine_consecutive_chats(chats, consecutive_time)
    
    topics = [[combined_chats[0]['content']]]
    topic_mapping = {combined_chat['id']: 0 for combined_chat in combined_chats[0]['original_chats']}
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
                topic_mapping[chat['id']] = -1
            continue

        print(f"Combined Chat {i}: Content: {content}")
        
        if current_speaker in last_speaker_info:
            last_time, last_topic = last_speaker_info[current_speaker]
            if (current_time - last_time).total_seconds() <= 60:
                assigned_topic = last_topic
                print(f"Combined Chat {i}: Same speaker within 1 minute, assigned to topic {assigned_topic + 1}")
            else:
                assigned_topic = assign_topic_with_ensemble_model(
                    nsp_model, topics, content, current_time, topic_times, combined_threshold
                )
        else:
            assigned_topic = assign_topic_with_ensemble_model(
                nsp_model, topics, content, current_time, topic_times, combined_threshold
            )
        
        if assigned_topic >= len(topics):
            topics.append([content])
            topic_times.append(current_time)
        else:
            topics[assigned_topic].append(content)
            if len(topics[assigned_topic]) > MAX_TOPIC_MESSAGES:
                topics[assigned_topic] = topics[assigned_topic][-MAX_TOPIC_MESSAGES:]
        
        for chat in combined_chat['original_chats']:
            topic_mapping[chat['id']] = assigned_topic
        
        topic_times[assigned_topic] = current_time
        last_speaker_info[current_speaker] = (current_time, assigned_topic)

    return topic_mapping, chats, topics, {}

@app.route('/predict', methods=['POST'])
async def predict():
    try:
        data = await request.get_json()
        channel_id = data.get('channelId')
        chats = data.get('chats', [])
        
        if not chats:
            return jsonify({'error': 'No chat messages provided'}), 400
            
        # Convert string dates to datetime objects
        for chat in chats:
            chat['createdAt'] = datetime.fromisoformat(chat['createdAt'].replace('Z', '+00:00'))
        
        # Process for each parameter set
        topics_per_param = {}
        summaries_per_param = {}        

        for param_set, params in PARAMETER_SETS.items():
            # Run topic classification logic with specific parameters
            topic_mapping, messages, topics, _ = await assign_topics_with_params(
                deepcopy(chats),
                params['combined_threshold'],
                params['consecutive_time']
            )
            
            # Count chats per topic
            topic_chat_counts = {}
            topics_for_summary = {}
            for chat_id, topic in topic_mapping.items():
                if topic not in topic_chat_counts:
                    topic_chat_counts[topic] = 0
                    topics_for_summary[topic] = []
                topic_chat_counts[topic] += 1
            
            # Select significant topics (4+ messages)
            significant_topics = {}
            for topic_id, count in topic_chat_counts.items():
                if count >= 4 and topic_id >= 0:
                    significant_topics[str(topic_id)] = topics[topic_id]
            
            # Summarize topics
            topic_summaries = await summarize_all_topics(significant_topics) if significant_topics else {}
            
            # Prepare result for this parameter set
            topics_per_param[param_set] = [topic_mapping.get(chat["id"], -1) for chat in chats]
            summaries_per_param[param_set] = topic_summaries
        
        return jsonify({"topics": topics_per_param, "summaries": summaries_per_param})
        
    except Exception as e:
        print(f"Error in predict: {str(e)}")
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
        traceback.print_exc()
