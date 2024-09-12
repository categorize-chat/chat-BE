from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
from openai import OpenAI
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId
from sklearn.model_selection import train_test_split

# .env 파일 로드
load_dotenv()

app = Flask(__name__)

# MongoDB 연결
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database_name']
chat_collection = db['chats']

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_catdflow_model(input_dim, lstm_units):
    model = keras.Sequential([
        keras.layers.LSTM(lstm_units, input_shape=(None, input_dim), return_sequences=True),
        keras.layers.Dense(1)
    ])
    return model

def create_catdmatch_model(input_dim):
    inputs = keras.Input(shape=(None, input_dim))
    lstm_output = keras.layers.LSTM(input_dim, return_sequences=True)(inputs)
    attention_output = keras.layers.Attention()([lstm_output, lstm_output])
    dense_output = keras.layers.Dense(input_dim, activation='tanh')(attention_output)
    model = keras.Model(inputs=inputs, outputs=dense_output)
    return model

def train_and_save_models(X_train, y_train, epochs=10):
    input_dim = X_train.shape[-1]
    lstm_units = 400

    catdflow_model = create_catdflow_model(input_dim, lstm_units)
    catdflow_model.compile(optimizer='adam', loss='mse')
    catdflow_model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)

    catdmatch_model = create_catdmatch_model(input_dim)
    catdmatch_model.compile(optimizer='adam', loss='mse')
    catdmatch_model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)

    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    catdflow_weights_path = os.path.join(models_dir, 'catdflow_model_weights.h5')
    catdmatch_weights_path = os.path.join(models_dir, 'catdmatch_model_weights.h5')
    
    catdflow_model.save_weights(catdflow_weights_path)
    catdmatch_model.save_weights(catdmatch_weights_path)

    return catdflow_model, catdmatch_model

def prepare_training_data(room_id):
    # 특정 채팅방의 마지막 100개 메시지 가져오기
    chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', -1).limit(100))
    chats.reverse()  # 시간 순으로 정렬

    if len(chats) < 10:  # 메시지가 10개 미만이면 에러 발생
        raise ValueError("Not enough messages in the chat room")

    # 채팅 메시지 전처리
    processed_chats = preprocess_chats(chats)

    X = []
    y = []

    # 학습 데이터 생성
    for i in range(1, len(processed_chats)):
        thread = processed_chats[:i]
        new_message = processed_chats[i]

        # 스레드의 마지막 20개 메시지만 사용
        thread = thread[-20:]

        X.append(np.array(thread + [new_message]))
        
        # 레이블 생성 (같은 스레드면 1, 아니면 0)
        time_diff = abs(chats[i]['createdAt'] - chats[i-1]['createdAt']).total_seconds()
        y.append(1 if time_diff <= 300 else 0)  # 5분 이내면 같은 스레드로 가정

    # numpy 배열로 변환
    X = np.array(X)
    y = np.array(y)

    # 학습 데이터와 검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_val, y_val

def preprocess_chats(chats):
    processed_chats = []
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

    for chat in chats:
        embedding = chat.get('embedding', [0] * 1536)  # 임베딩이 없는 경우 0으로 채움
        nickname_emb = get_nickname_embedding(chat['nickname']) / len(nickname_map)
        time_diff = get_time_diff_embedding(chat['createdAt'], datetime.now())
        
        processed_chat = embedding + [nickname_emb, time_diff / 11]
        processed_chats.append(processed_chat)

    return processed_chats

# 모델 가중치 파일 경로 설정
models_dir = os.path.join(os.path.dirname(__file__), 'models')
catdflow_weights_path = os.path.join(models_dir, 'catdflow_model_weights.h5')
catdmatch_weights_path = os.path.join(models_dir, 'catdmatch_model_weights.h5')

# 모델 생성 또는 로드
input_dim = 1538  # 임베딩 차원(1536) + 추가 특성(2)
lstm_units = 400

if os.getenv('TRAIN_MODELS', 'false').lower() == 'true':
    room_id = os.getenv('TRAIN_ROOM_ID')
    if not room_id:
        raise ValueError("TRAIN_ROOM_ID must be set in .env file for training")
    X_train, y_train, _, _ = prepare_training_data(room_id)
    catdflow_model, catdmatch_model = train_and_save_models(X_train, y_train)
else:
    catdflow_model = create_catdflow_model(input_dim, lstm_units)
    catdmatch_model = create_catdmatch_model(input_dim)
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