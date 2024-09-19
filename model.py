from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime, timezone
from openai import OpenAI
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId
from dateutil import parser

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

# 연결 테스트
try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    raise

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 전역 변수로 임베딩 차원 설정
EMBEDDING_DIM = 1536  # OpenAI의 text-embedding-3-small 모델의 실제 출력 차원
ADDITIONAL_FEATURES = 2  # nickname_emb와 time_diff
EXPECTED_DIM = EMBEDDING_DIM + ADDITIONAL_FEATURES

def create_catdflow_model(lstm_units, output_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(None, EXPECTED_DIM)),
        keras.layers.LSTM(lstm_units, return_sequences=False),
        keras.layers.Dense(output_dim)
    ])
    return model

def create_catdmatch_model(output_dim):
    inputs = keras.Input(shape=(None, EXPECTED_DIM))
    lstm_output = keras.layers.LSTM(EXPECTED_DIM, return_sequences=False)(inputs)
    dense_output = keras.layers.Dense(output_dim)(lstm_output)
    model = keras.Model(inputs=inputs, outputs=dense_output)
    return model

def get_or_create_embedding(chat_id, content):
    chat_doc = chat_collection.find_one({"_id": chat_id})
    if chat_doc and 'embedding' in chat_doc:
        return chat_doc['embedding']
    
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=content
    )
    embedding = response.data[0].embedding
    
    chat_collection.update_one(
        {"_id": chat_id},
        {"$set": {"embedding": embedding}},
        upsert=True
    )
    
    return embedding

def preprocess_input(thread, new_message):
    try:
        nickname_map = {}
        
        def get_nickname_embedding(nickname):
            if nickname not in nickname_map:
                nickname_map[nickname] = len(nickname_map)
            return nickname_map[nickname]
        
        def get_time_diff_embedding(time1, time2):
            if time1.tzinfo is None:
                time1 = time1.replace(tzinfo=timezone.utc)
            if time2.tzinfo is None:
                time2 = time2.replace(tzinfo=timezone.utc)
            
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
        
        current_time = datetime.now(timezone.utc)
        
        processed_thread = []
        processed_ids = set()  # 중복 검사를 위한 set

        if isinstance(thread, list) and len(thread) > 0 and isinstance(thread[0], list):
            thread = thread[0]

        for msg in thread:
            msg_id = msg.get('id') or msg.get('_id')
            if not msg_id:
                print(f"Warning: Message without id: {msg}")
                continue
            
            if msg_id in processed_ids:
                continue
            processed_ids.add(msg_id)
            
            embedding = get_or_create_embedding(msg_id, msg['content'])
            nickname_emb = get_nickname_embedding(msg['nickname']) / (len(nickname_map) + 1)
            
            if isinstance(msg['createdAt'], datetime):
                msg_time = msg['createdAt']
            else:
                msg_time = parser.isoparse(msg['createdAt'])
            
            if msg_time.tzinfo is None:
                msg_time = msg_time.replace(tzinfo=timezone.utc)
            
            time_diff = get_time_diff_embedding(msg_time, current_time)
            processed_thread.append(embedding + [nickname_emb, time_diff / 11])
        
        new_msg_id = new_message.get('_id') or new_message.get('id')
        new_msg_embedding = get_or_create_embedding(new_msg_id, new_message['content'])
        new_msg_nickname_emb = get_nickname_embedding(new_message['nickname']) / (len(nickname_map) + 1)
        
        if isinstance(new_message['createdAt'], datetime):
            new_msg_time = new_message['createdAt']
        else:
            new_msg_time = parser.isoparse(new_message['createdAt'])
        
        if new_msg_time.tzinfo is None:
            new_msg_time = new_msg_time.replace(tzinfo=timezone.utc)
        
        new_msg_time_diff = get_time_diff_embedding(new_msg_time, current_time)
        processed_new_message = new_msg_embedding + [new_msg_nickname_emb, new_msg_time_diff / 11]
        
        processed_thread = np.array(processed_thread)
        processed_new_message = np.array(processed_new_message)

        if len(processed_thread) > 0 and processed_thread.shape[1] != EXPECTED_DIM:
            print(f"Warning: Thread dimension mismatch. Expected {EXPECTED_DIM}, got {processed_thread.shape[1]}")
            if processed_thread.shape[1] > EXPECTED_DIM:
                processed_thread = processed_thread[:, :EXPECTED_DIM]
            else:
                padding = np.zeros((processed_thread.shape[0], EXPECTED_DIM - processed_thread.shape[1]))
                processed_thread = np.hstack((processed_thread, padding))

        if len(processed_new_message) != EXPECTED_DIM:
            print(f"Warning: New message dimension mismatch. Expected {EXPECTED_DIM}, got {len(processed_new_message)}")
            if len(processed_new_message) > EXPECTED_DIM:
                processed_new_message = processed_new_message[:EXPECTED_DIM]
            else:
                processed_new_message = np.pad(processed_new_message, (0, EXPECTED_DIM - len(processed_new_message)), 'constant')

        if len(processed_thread) == 0:
            print("processed_thread is empty. Using only new_message.")
            processed_thread = np.array([processed_new_message])

        return processed_thread, processed_new_message
    except Exception as e:
        print(f"Error in preprocess_input: {str(e)}")
        print("new_message causing error:", new_message)
        print("thread:", thread)
        raise

def prepare_training_data(room_id):
    chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', -1).limit(100))
    
    if len(chats) < 2:
        raise ValueError(f"Not enough messages in the chat room. Found {len(chats)} messages.")

    X = []
    y = []
    
    for i in range(1, len(chats)):
        thread = chats[:i]
        new_message = chats[i]
        
        thread_processed, new_message_processed = preprocess_input(thread, new_message)
        
        max_thread_length = 20
        if len(thread_processed) > max_thread_length:
            thread_processed = thread_processed[-max_thread_length:]
        else:
            padding = np.zeros((max_thread_length - len(thread_processed), thread_processed.shape[1]))
            thread_processed = np.vstack((padding, thread_processed))
        
        X.append(thread_processed)
        y.append(new_message_processed)
    
    return np.array(X), np.array(y), len(chats), len(chats) - 1

def train_and_save_models(X_train, y_train, catdflow_model, catdmatch_model, epochs=10):
    catdflow_model.compile(optimizer='adam', loss='mse')
    catdflow_model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)

    catdmatch_model.compile(optimizer='adam', loss='mse')
    catdmatch_model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)

    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    catdflow_model_path = os.path.join(models_dir, 'catdflow_model.keras')
    catdmatch_model_path = os.path.join(models_dir, 'catdmatch_model.keras')
    
    catdflow_model.save(catdflow_model_path)
    catdmatch_model.save(catdmatch_model_path)

    print(f"Models saved to {models_dir}")

    return catdflow_model, catdmatch_model

def load_or_create_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    catdflow_model_path = os.path.join(models_dir, 'catdflow_model.keras')
    catdmatch_model_path = os.path.join(models_dir, 'catdmatch_model.keras')

    if os.path.exists(catdflow_model_path) and os.path.exists(catdmatch_model_path):
        print("Loading existing models...")
        catdflow_model = keras.models.load_model(catdflow_model_path)
        catdmatch_model = keras.models.load_model(catdmatch_model_path)
        print("Models loaded successfully")
    else:
        print("Creating new models...")
        lstm_units = 400
        catdflow_model = create_catdflow_model(lstm_units, EXPECTED_DIM)
        catdmatch_model = create_catdmatch_model(EXPECTED_DIM)
        
        room_id = '66b0fd658aab9f2bd7a41845'
        X_train, y_train, total_messages, training_samples = prepare_training_data(room_id)
        print(f"Total messages in room: {total_messages}")
        print(f"Training samples prepared: {training_samples}")
        
        if training_samples > 0:
            catdflow_model, catdmatch_model = train_and_save_models(X_train, y_train, catdflow_model, catdmatch_model)
            print("Models trained and saved successfully")
        else:
            print("No training samples could be prepared. Check your data.")

    return catdflow_model, catdmatch_model

def combine_predictions(flow_prediction, match_prediction):
    g = 1 / (1 + np.exp(-match_prediction))
    return (1 - g) * match_prediction + g * flow_prediction

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        thread = data['thread']
        new_message = data['new_message']

        if not thread:
            print("Thread is empty. Using only new_message for prediction.")
            thread = [new_message]

        if '_id' not in new_message:
            new_message['_id'] = str(ObjectId())

        try:
            preprocessed_thread, preprocessed_new_message = preprocess_input(thread, new_message)
        except Exception as e:
            print(f"Error in preprocess_input: {str(e)}")
            return jsonify({'error': str(e)}), 500

        max_thread_length = 20
        if len(preprocessed_thread) > max_thread_length:
            preprocessed_thread = preprocessed_thread[-max_thread_length:]
        elif len(preprocessed_thread) < max_thread_length:
            padding = np.zeros((max_thread_length - len(preprocessed_thread), preprocessed_thread.shape[1]))
            preprocessed_thread = np.vstack((preprocessed_thread, padding))

        flow_input = np.expand_dims(preprocessed_thread, 0)
        
        try:
            flow_prediction = catdflow_model.predict(flow_input).squeeze()
            match_prediction = catdmatch_model.predict(flow_input).squeeze()
        except Exception as e:
            print(f"Error in model prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500

        combined_prediction = combine_predictions(flow_prediction, match_prediction)

        return jsonify({
            'prediction': combined_prediction.tolist()
        })
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    catdflow_model, catdmatch_model = load_or_create_models()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"Server started on port {port}")
    app.run(host='0.0.0.0', port=port)
