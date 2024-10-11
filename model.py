from flask import Flask, request, jsonify, Response
import json
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
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report

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

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 전역 변수 설정
EMBEDDING_DIM = 1536
USER_EMB_DIM = 64
TIME_EMB_DIM = 64
ADDITIONAL_FEATURES = USER_EMB_DIM + TIME_EMB_DIM
EXPECTED_DIM = EMBEDDING_DIM + ADDITIONAL_FEATURES
INITIAL_NUM_CLASSES = 5
BATCH_SIZE = 32
MAX_THREAD_LENGTH = 20
SIMILARITY_THRESHOLD = 0.3
MAX_THREADS = 15

embedding_cache = {}

class ImprovedCATD(keras.Model):
    def __init__(self, lstm_units=256, attention_units=128, output_dim=EXPECTED_DIM, initial_num_classes=INITIAL_NUM_CLASSES, dropout_rate=0.5):
        super(ImprovedCATD, self).__init__()
        self.lstm_units = lstm_units
        self.attention_units = attention_units

        self.user_embedding_layer = keras.layers.Embedding(input_dim=1000, output_dim=USER_EMB_DIM)
        self.time_embedding_layer = keras.layers.Dense(TIME_EMB_DIM, activation='relu')
        
        self.thread_dense = keras.layers.Dense(self.lstm_units, activation='relu')
        self.thread_lstm = keras.layers.LSTM(self.lstm_units, return_sequences=True)
        
        self.message_dense = keras.layers.Dense(self.lstm_units, activation='relu')
        
        # Add these layers to adjust dimensions for attention
        self.thread_attention_dense = keras.layers.Dense(self.attention_units, activation='relu')
        self.message_attention_dense = keras.layers.Dense(self.attention_units, activation='relu')
        self.time_attention_dense = keras.layers.Dense(self.attention_units, activation='relu')
        
        self.attention = keras.layers.Attention()
        self.time_attention = keras.layers.Attention()
        self.normalization = keras.layers.LayerNormalization()
        
        self.dense1 = keras.layers.Dense(output_dim, activation='relu')
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.classifier = keras.layers.Dense(initial_num_classes, activation='softmax')

    def call(self, inputs, training=False):
        thread, message, user_ids, time_diffs = inputs
        
        # Thread processing
        thread_processed = self.thread_dense(thread)
        user_emb_thread = self.user_embedding_layer(user_ids)
        time_emb_thread = self.time_embedding_layer(time_diffs)
        
        combined_thread = tf.concat([thread_processed, user_emb_thread, time_emb_thread], axis=-1)
        thread_output = self.thread_lstm(combined_thread)
        
        # Message processing
        message_output = self.message_dense(message)
        message_output = tf.expand_dims(message_output, 1)
        
        # Adjust dimensions for attention
        thread_attention = self.thread_attention_dense(thread_output)
        message_attention = self.message_attention_dense(message_output)
        time_attention = self.time_attention_dense(time_emb_thread)
        
        # Attention
        context_vector = self.attention([message_attention, thread_attention])
        
        # Time attention
        time_context = self.time_attention([message_attention, time_attention])
        
        # Combine context vectors
        combined_context = tf.concat([context_vector, time_context], axis=-1)
        combined_context = tf.squeeze(combined_context, 1)
        
        # Normalization and final layers
        normalized = self.normalization(combined_context)
        x = self.dense1(normalized)
        x = self.dropout(x, training=training)
        return self.classifier(x)

    def expand_classifier(self, new_num_classes):
        old_weights, old_biases = self.classifier.get_weights()
        new_weights = np.pad(old_weights, ((0, 0), (0, new_num_classes - self.classifier.units)))
        new_biases = np.pad(old_biases, (0, new_num_classes - self.classifier.units))
        self.classifier = keras.layers.Dense(new_num_classes, activation='softmax')
        self.classifier.set_weights([new_weights, new_biases])

def get_or_create_embedding(chat_id, content):
    if chat_id in embedding_cache:
        return embedding_cache[chat_id]
    
    chat_doc = chat_collection.find_one({"_id": chat_id}, {"embedding": 1})
    if chat_doc and 'embedding' in chat_doc:
        embedding = chat_doc['embedding']
    else:
        print("Creating new embedding")
        try:
            response = openai_client.embeddings.create(model="text-embedding-3-small", input=content)
            embedding = response.data[0].embedding
            chat_collection.update_one({"_id": chat_id}, {"$set": {"embedding": embedding}}, upsert=True)
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            embedding = np.random.rand(EMBEDDING_DIM)  # 임시 대체 임베딩
    
    embedding_cache[chat_id] = embedding
    return embedding

def safe_int_convert(value, default=1):
    try:
        return int(float(value))  # float로 변환 후 int로 변환
    except (ValueError, TypeError):
        return default
    
def get_normalized_topic(original_topic):
    return safe_int_convert(original_topic, default=-1)

def assign_topics(room_id, max_topics=MAX_THREADS, min_topic_size=2):
    print("Starting assign_topics function")
    
    chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1))
    topics = []
    current_topic = []
    topic_mapping = {}

    for i, chat in enumerate(chats):
        chat_id = chat['_id']
        content = chat['content']

        print(f"Processing chat {i}: Content: {content[:50]}...")

        embedding = get_or_create_embedding(chat_id, content)
        
        if not current_topic:
            current_topic.append((chat_id, embedding))
            topic_mapping[str(chat_id)] = 1  # Start with topic 1
            print(f"Chat {i}: First message in new topic 1")
        else:
            similarities = [cosine_similarity([embedding], [e])[0][0] for _, e in current_topic]
            max_similarity = max(similarities)
            
            if max_similarity < SIMILARITY_THRESHOLD and len(topics) < max_topics - 1:
                if len(current_topic) >= min_topic_size:
                    topics.append(current_topic)
                    current_topic = [(chat_id, embedding)]
                    topic_mapping[str(chat_id)] = len(topics) + 1
                    print(f"Chat {i}: Low similarity ({max_similarity:.4f}), created new topic {len(topics) + 1}")
                else:
                    topic_mapping[str(chat_id)] = 0  # 아무 토픽에도 속하지 않는 메시지는 0으로 표시
                    print(f"Chat {i}: Low similarity but current topic too small, assigned 0")
            else:
                current_topic.append((chat_id, embedding))
                topic_mapping[str(chat_id)] = len(topics) + 1
                print(f"Chat {i}: High similarity or max topics reached, assigned to existing topic {len(topics) + 1}")

        print(f"Current unique topics: {set(topic_mapping.values())}")
        print(f"Assigned topic: {topic_mapping[str(chat_id)]}")
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(chats)} chats")
            print(f"Current topic distribution: {Counter(topic_mapping.values())}")

    if current_topic and len(current_topic) >= min_topic_size:
        topics.append(current_topic)
    elif current_topic:
        for chat_id, _ in current_topic:
            topic_mapping[str(chat_id)] = 0

    print(f"Assigned topics to {len(topic_mapping)} messages")
    print(f"Final topic distribution: {Counter(topic_mapping.values())}")
    
    return topic_mapping

def preprocess_input(thread, new_message=None):
    nickname_map = {}
    
    def get_nickname_id(nickname):
        if nickname not in nickname_map:
            nickname_map[nickname] = len(nickname_map)
        return nickname_map[nickname]
    
    def get_time_diff(current_time, thread_time):
        diff_minutes = abs((current_time - thread_time).total_seconds()) / 60
        return min(diff_minutes / 30, 1)  # Normalize to [0, 1] with max of 24 hours
    
    processed_thread = []
    user_ids = []
    time_diffs = []
    
    full_thread = thread + [new_message] if new_message else thread
    full_thread = full_thread[-MAX_THREAD_LENGTH:]  # Consider only the last MAX_THREAD_LENGTH messages
    
    for i, current_msg in enumerate(full_thread):
        msg_id = current_msg.get('id') or current_msg.get('_id')
        embedding = get_or_create_embedding(msg_id, current_msg['content'])
        user_id = get_nickname_id(current_msg['nickname'])
        current_time = current_msg['createdAt'] if isinstance(current_msg['createdAt'], datetime) else parser.isoparse(current_msg['createdAt'])
        
        # Calculate time differences with all previous messages in the thread
        msg_time_diffs = []
        for prev_msg in full_thread[:i]:
            prev_time = prev_msg['createdAt'] if isinstance(prev_msg['createdAt'], datetime) else parser.isoparse(prev_msg['createdAt'])
            msg_time_diffs.append(get_time_diff(current_time, prev_time))
        
        # Pad time_diffs if necessary
        msg_time_diffs = [0.0] * (MAX_THREAD_LENGTH - 1 - len(msg_time_diffs)) + msg_time_diffs
        
        processed_thread.append(embedding)
        user_ids.append(user_id)
        time_diffs.append(msg_time_diffs)
    
    # Padding for the entire thread if necessary
    while len(processed_thread) < MAX_THREAD_LENGTH:
        processed_thread.append([0.0] * EMBEDDING_DIM)
        user_ids.append(0)
        time_diffs.append([0.0] * (MAX_THREAD_LENGTH - 1))
    
    time_diffs = np.array(time_diffs)
    if time_diffs.ndim == 3:
        time_diffs = time_diffs.mean(axis=2)  # 마지막 차원의 평균을 취합니다

    return np.array(processed_thread), np.array(user_ids), time_diffs

def data_generator(room_id, batch_size=BATCH_SIZE):
    chats = chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1)
    total_chats = chat_collection.count_documents({'room': ObjectId(room_id)})
    
    batch_thread, batch_message, batch_user, batch_time, batch_y = [], [], [], [], []
    topic_to_index = {i: i for i in range(MAX_THREADS)}
    
    for i, chat in enumerate(chats):
        if i == 0:
            continue
        
        thread = list(chat_collection.find({'room': ObjectId(room_id), 'createdAt': {'$lt': chat['createdAt']}}).sort('createdAt', -1).limit(MAX_THREAD_LENGTH - 1))
        thread.reverse()
        
        try:
            thread_embed, user_ids, time_diffs = preprocess_input(thread, chat)
            message_embed = get_or_create_embedding(chat['_id'], chat['content'])
            
            topic = min(max(int(chat.get('topic', 0)), 0), MAX_THREADS - 1)
            y = topic_to_index[topic]
            
            batch_thread.append(thread_embed)
            batch_message.append(message_embed)
            batch_user.append(user_ids)
            batch_time.append(time_diffs)
            batch_y.append(y)
            
            if len(batch_thread) == batch_size:
                yield [np.array(batch_thread), np.array(batch_message), np.array(batch_user), np.array(batch_time)], np.array(batch_y)
                batch_thread, batch_message, batch_user, batch_time, batch_y = [], [], [], [], []
        
        except Exception as e:
            print(f"Error processing chat {i}: {str(e)}")
        
        if i % 100 == 0:
            print(f"Processed {i}/{total_chats} chats...")
    
    if batch_thread:
        yield [np.array(batch_thread), np.array(batch_message), np.array(batch_user), np.array(batch_time)], np.array(batch_y)  

def train_model(room_id, epochs=10):
    print("Starting improved model training...")
    num_classes = MAX_THREADS  # 클래스 수를 MAX_THREADS로 설정
    
    model = ImprovedCATD(lstm_units=128, output_dim=EXPECTED_DIM, initial_num_classes=num_classes, dropout_rate=0.5)
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    total_samples = chat_collection.count_documents({'room': ObjectId(room_id)}) - 1
    steps_per_epoch = total_samples // BATCH_SIZE
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        generator = data_generator(room_id)
        model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=1, verbose=1)
    
    print("Improved model training completed")
    return model

def save_model(model):
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'improved_catd_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

def evaluate_predictions(true_topics, predicted_topics):
    true_labels = [true_topics.get(chat_id, -1) for chat_id in predicted_topics.keys()]
    pred_labels = list(predicted_topics.values())
    
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, zero_division=0)
    
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(report)
    
    return accuracy, report

# 모델 초기화
model = ImprovedCATD(lstm_units=256, attention_units=128, output_dim=EXPECTED_DIM, initial_num_classes=MAX_THREADS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        room_id = data['room_id']
        
        print("Room ID:", room_id)
        
        chat_count = chat_collection.count_documents({'room': ObjectId(room_id)})
        print(f"Number of chat messages for room {room_id}: {chat_count}")
        
        if chat_count == 0:
            return jsonify({'error': 'No chat messages found for this room'}), 404
        
        chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1))
        result = []
        
        for i, chat in enumerate(chats[1:]):  # Skip the first message
            thread = chats[:i+1]
            thread_embed, user_ids, time_diffs = preprocess_input(thread[:-1], chat)
            message_embed = get_or_create_embedding(chat['_id'], chat['content'])
            
            inputs = [
                np.expand_dims(thread_embed, 0),
                np.expand_dims(message_embed, 0),
                np.expand_dims(user_ids, 0),
                np.expand_dims(time_diffs, 0)
            ]
            
            prediction = model.predict(inputs)
            predicted_topic = int(np.argmax(prediction[0]))
            
            result.append({
                'content': chat['content'],
                'predicted_topic': predicted_topic
            })
        
        return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json')
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()   
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting improved CATD model...")
    room_id = '66b0fd658aab9f2bd7a41841'
    
    try:
        model = train_model(room_id)
        save_model(model)
        
        port = int(os.environ.get('PORT', 5000))
        print(f"Server started on port {port}")
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")