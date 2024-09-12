from dotenv import load_dotenv
import os

load_dotenv()

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
from openai import OpenAI
from pymongo import MongoClient

app = Flask(__name__)

# MongoDB 연결
client = MongoClient('mongodb://localhost:27017/')
db = client['aichat']
chat_collection = db['chats']

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CATDFLOW 모델 정의
def create_catdflow_model(input_dim, lstm_units):
    model = keras.Sequential([
        keras.layers.LSTM(lstm_units, input_shape=(None, input_dim), return_sequences=True),
        keras.layers.Dense(1)
    ])
    return model

# CATDMATCH 모델 정의
def create_catdmatch_model(input_dim):
    inputs = keras.Input(shape=(None, input_dim))
    lstm_output = keras.layers.LSTM(input_dim, return_sequences=True)(inputs)
    attention_output = keras.layers.Attention()([lstm_output, lstm_output])
    dense_output = keras.layers.Dense(input_dim, activation='tanh')(attention_output)
    model = keras.Model(inputs=inputs, outputs=dense_output)
    return model

# 모델 가중치 파일 경로 설정
models_dir = os.path.join(os.path.dirname(__file__), 'models')
catdflow_weights_path = os.path.join(models_dir, 'catdflow_model_weights.h5')
catdmatch_weights_path = os.path.join(models_dir, 'catdmatch_model_weights.h5')

# 모델 생성
input_dim = 1576  # 임베딩 차원 + 추가 특성
lstm_units = 400
catdflow_model = create_catdflow_model(input_dim, lstm_units)
catdmatch_model = create_catdmatch_model(input_dim)

# 모델 가중치 로드
catdflow_model.load_weights(catdflow_weights_path)
catdmatch_model.load_weights(catdmatch_weights_path)

def get_or_create_embedding(chat_id, content):
    # MongoDB에서 임베딩 검색
    chat_doc = chat_collection.find_one({"_id": chat_id})
    if chat_doc and 'embedding' in chat_doc:
        return chat_doc['embedding']
    
    # 임베딩이 없는 경우, 새로 생성
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=content
    )
    embedding = response.data[0].embedding
    
    # 새 임베딩을 MongoDB에 저장
    chat_collection.update_one(
        {"_id": chat_id},
        {"$set": {"embedding": embedding}},
        upsert=True
    )
    
    return embedding

def preprocess_input(thread, new_message):
    nickname_map = {}
    
    def get_nickname_embedding(nickname):
        if nickname not in nickname_map:
            nickname_map[nickname] = len(nickname_map)
        return nickname_map[nickname]
    
    def get_time_diff_embedding(time1, time2):
        diff_minutes = abs((time1 - time2).total_seconds()) / 60
        if diff_minutes <= 1: return 0
        elif diff_minutes <= 5: return 1
        elif diff_minutes <= 10: return 2
        elif diff_minutes <= 30: return 3
        elif diff_minutes <= 60: return 4
        elif diff_minutes <= 120: return 5
        elif diff_minutes <= 240: return 6
        elif diff_minutes <= 480: return 7
        elif diff_minutes <= 720: return 8
        elif diff_minutes <= 1440: return 9
        else: return 10
    
    current_time = datetime.now()
    
    # thread 전처리
    processed_thread = []
    for msg in thread:
        embedding = get_or_create_embedding(msg['id'], msg['content'])
        nickname_emb = get_nickname_embedding(msg['nickname']) / len(nickname_map)
        time_diff = get_time_diff_embedding(datetime.fromisoformat(msg['createdAt']), current_time)
        processed_thread.append(embedding + [nickname_emb, time_diff / 11])
    
    # new_message 전처리
    new_msg_embedding = get_or_create_embedding(new_message['id'], new_message['content'])
    new_msg_nickname_emb = get_nickname_embedding(new_message['nickname']) / len(nickname_map)
    processed_new_message = new_msg_embedding + [new_msg_nickname_emb, 0]  # 시간 차이는 0으로 설정
    
    return np.array(processed_thread), np.array(processed_new_message)

def combine_predictions(flow_prediction, match_prediction):
    g = 1 / (1 + np.exp(-match_prediction))
    return (1 - g) * match_prediction + g * flow_prediction

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    thread = data['thread']
    new_message = data['new_message']

    preprocessed_thread, preprocessed_message = preprocess_input(thread, new_message)

    flow_input = np.concatenate([preprocessed_thread, np.expand_dims(preprocessed_message, 0)], axis=0)
    flow_input = np.expand_dims(flow_input, 0)
    flow_prediction = catdflow_model.predict(flow_input).squeeze()

    match_prediction = catdmatch_model.predict(np.expand_dims(preprocessed_thread, 0))
    match_prediction = tf.keras.losses.cosine_similarity(match_prediction, np.expand_dims(preprocessed_message, 0)).numpy()

    combined_prediction = combine_predictions(flow_prediction, match_prediction)

    return jsonify({
        'prediction': combined_prediction.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)