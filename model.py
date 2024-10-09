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
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, classification_report

load_dotenv()

app = Flask(__name__)

mongodb_uri = os.getenv('MONGODB_URI')
if not mongodb_uri or not mongodb_uri.startswith(('mongodb://', 'mongodb+srv://')):
    raise ValueError("Invalid MONGODB_URI. It must start with 'mongodb://' or 'mongodb+srv://'")

client = MongoClient(mongodb_uri)
db = client.get_database()
chat_collection = db['chats']

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_DIM = 1536
USER_EMB_DIM = 64
TIME_EMB_DIM = 64
ADDITIONAL_FEATURES = USER_EMB_DIM + TIME_EMB_DIM
EXPECTED_DIM = EMBEDDING_DIM + ADDITIONAL_FEATURES
INITIAL_NUM_CLASSES = 5
BATCH_SIZE = 32
MAX_THREAD_LENGTH = 20
SIMILARITY_THRESHOLD = 0.35  # 임계값 낮춤
MAX_THREADS = 7  # 최대 토픽 수 증가
CONTEXT_WINDOW = 5  # 컨텍스트 윈도우 크기

embedding_cache = {}

class EnhancedDynamicClassifier(keras.Model):
    def __init__(self, lstm_units, output_dim, initial_num_classes, dropout_rate=0.5):
        super(EnhancedDynamicClassifier, self).__init__()
        self.text_embedding_layer = keras.layers.Dense(EMBEDDING_DIM, activation='relu')
        self.user_embedding_layer = keras.layers.Embedding(input_dim=1000, output_dim=USER_EMB_DIM)
        self.time_embedding_layer = keras.layers.Dense(TIME_EMB_DIM, activation='relu')
        
        self.lstm1 = keras.layers.LSTM(lstm_units, return_sequences=True)
        self.lstm2 = keras.layers.LSTM(lstm_units, return_sequences=True)
        self.attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=lstm_units)
        self.flatten = keras.layers.Flatten()
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dense1 = keras.layers.Dense(output_dim, activation='relu')
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.dense2 = keras.layers.Dense(output_dim // 2, activation='relu')
        self.classifier = keras.layers.Dense(initial_num_classes, activation='softmax')

    def call(self, inputs, training=False):
        text_emb, user_ids, time_diffs = inputs
        
        user_emb = self.user_embedding_layer(user_ids)
        time_diffs_expanded = tf.expand_dims(time_diffs, -1)
        time_emb = self.time_embedding_layer(time_diffs_expanded)
        
        combined_emb = tf.concat([text_emb, user_emb, time_emb], axis=-1)
        
        x = self.lstm1(combined_emb)
        x = self.lstm2(x)
        x = self.attention(x, x)
        x = self.flatten(x)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
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
            embedding = np.random.rand(EMBEDDING_DIM)
    
    embedding_cache[chat_id] = embedding
    return embedding

def safe_int_convert(value, default=1):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def assign_topics(room_id):
    print("Starting assign_topics function")
    
    chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1))
    predicted_topics = {}
    topic_embeddings = []
    
    for i, chat in enumerate(chats):
        chat_id = chat['_id']
        try:
            embedding = get_or_create_embedding(chat_id, chat['content'])
            
            print(f"Chat {i}: Content: {chat['content'][:50]}...")
            
            if not topic_embeddings:
                predicted_topic = 1
                print(f"Chat {i}: First message, assigned to topic {predicted_topic}")
            else:
                # 컨텍스트 윈도우 적용
                context_embeddings = topic_embeddings[-CONTEXT_WINDOW:]
                similarities = cosine_similarity([embedding], context_embeddings)[0]
                max_similarity = np.max(similarities)
                
                if max_similarity >= SIMILARITY_THRESHOLD:
                    predicted_topic = predicted_topics[list(predicted_topics.keys())[-1]]
                    print(f"Chat {i}: High similarity ({max_similarity:.4f}), assigned to existing topic {predicted_topic}")
                else:
                    if len(set(predicted_topics.values())) < MAX_THREADS:
                        predicted_topic = len(set(predicted_topics.values())) + 1
                        print(f"Chat {i}: Low similarity ({max_similarity:.4f}), created new topic {predicted_topic}")
                    else:
                        # DBSCAN 클러스터링 적용
                        clusterer = DBSCAN(eps=0.5, min_samples=2)
                        cluster_labels = clusterer.fit_predict(np.array(topic_embeddings))
                        if -1 in cluster_labels:  # 새로운 클러스터 생성
                            predicted_topic = len(set(cluster_labels)) + 1
                        else:
                            predicted_topic = cluster_labels[-1] + 1
                        print(f"Chat {i}: Low similarity ({max_similarity:.4f}), assigned to topic {predicted_topic} based on clustering")
            
            predicted_topic = min(safe_int_convert(predicted_topic), MAX_THREADS)
            predicted_topics[str(chat_id)] = predicted_topic
            topic_embeddings.append(embedding)
            
            print(f"Current unique topics: {set(predicted_topics.values())}")
            print(f"Assigned topic: {predicted_topic}")
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(chats)} chats")
                print(f"Current topic distribution: {Counter(predicted_topics.values())}")
        
        except Exception as e:
            print(f"Error processing chat {i}: {str(e)}")
            continue
    
    # 토픽 병합 및 분할
    merge_and_split_topics(predicted_topics, topic_embeddings)
    
    print(f"Assigned topics to {len(predicted_topics)} messages")
    print(f"Final topic distribution: {Counter(predicted_topics.values())}")
    return predicted_topics

def merge_and_split_topics(predicted_topics, topic_embeddings):
    # 토픽 병합
    topic_centroids = {}
    for topic, embedding in zip(predicted_topics.values(), topic_embeddings):
        if topic not in topic_centroids:
            topic_centroids[topic] = []
        topic_centroids[topic].append(embedding)
    
    for topic, embeddings in topic_centroids.items():
        centroid = np.mean(embeddings, axis=0)
        topic_centroids[topic] = centroid
    
    merged_topics = {}
    for topic1, centroid1 in topic_centroids.items():
        for topic2, centroid2 in topic_centroids.items():
            if topic1 < topic2:
                similarity = cosine_similarity([centroid1], [centroid2])[0][0]
                if similarity > 0.8:  # 높은 유사도를 가진 토픽 병합
                    merged_topics[topic2] = topic1
    
    # 토픽 분할
    for topic, embeddings in topic_centroids.items():
        if len(embeddings) > 100:  # 큰 토픽 분할
            clusterer = DBSCAN(eps=0.3, min_samples=5)
            sub_labels = clusterer.fit_predict(embeddings)
            for i, (chat_id, old_topic) in enumerate(predicted_topics.items()):
                if old_topic == topic:
                    predicted_topics[chat_id] = f"{topic}-{sub_labels[i]}"
    
    # 병합된 토픽 적용
    for chat_id, topic in predicted_topics.items():
        if topic in merged_topics:
            predicted_topics[chat_id] = merged_topics[topic]

def preprocess_input(thread, new_message=None):
    nickname_map = {}
    current_time = datetime.now(timezone.utc)
    
    def get_nickname_id(nickname):
        if nickname not in nickname_map:
            nickname_map[nickname] = len(nickname_map)
        return nickname_map[nickname]
    
    def get_time_diff(msg_time):
        diff_minutes = abs((current_time - msg_time).total_seconds()) / 60
        return min(diff_minutes / 1440, 1)  # Normalize to [0, 1] with max of 24 hours
    
    processed_thread = []
    user_ids = []
    time_diffs = []
    
    full_thread = thread + [new_message] if new_message else thread
    
    for msg in full_thread[-MAX_THREAD_LENGTH:]:
        msg_id = msg.get('id') or msg.get('_id')
        embedding = get_or_create_embedding(msg_id, msg['content'])
        user_id = get_nickname_id(msg['nickname'])
        msg_time = msg['createdAt'] if isinstance(msg['createdAt'], datetime) else parser.isoparse(msg['createdAt'])
        time_diff = get_time_diff(msg_time.replace(tzinfo=timezone.utc))
        
        processed_thread.append(embedding)
        user_ids.append(user_id)
        time_diffs.append(time_diff)
    
    if len(processed_thread) < MAX_THREAD_LENGTH:
        padding_length = MAX_THREAD_LENGTH - len(processed_thread)
        processed_thread = [[0.0] * EMBEDDING_DIM] * padding_length + processed_thread
        user_ids = [0] * padding_length + user_ids
        time_diffs = [0.0] * padding_length + time_diffs
    
    return np.array(processed_thread), np.array(user_ids), np.array(time_diffs)

def data_generator(room_id, batch_size=BATCH_SIZE):
    chats = chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1)
    total_chats = chat_collection.count_documents({'room': ObjectId(room_id)})
    
    batch_x_text, batch_x_user, batch_x_time, batch_y = [], [], [], []
    topic_to_index = {i+1: i for i in range(MAX_THREADS)}
    
    for i, chat in enumerate(chats):
        if i == 0:
            continue
        
        thread = list(chat_collection.find({'room': ObjectId(room_id), 'createdAt': {'$lt': chat['createdAt']}}).sort('createdAt', -1).limit(MAX_THREAD_LENGTH - 1))
        thread.reverse()
        
        try:
            x_text, x_user, x_time = preprocess_input(thread, chat)
            topic = min(int(chat.get('topic', 1)), MAX_THREADS)
            y = topic_to_index[topic]
            
            batch_x_text.append(x_text)
            batch_x_user.append(x_user)
            batch_x_time.append(x_time)
            batch_y.append(y)
            
            if len(batch_x_text) == batch_size:
                yield [np.array(batch_x_text), np.array(batch_x_user), np.array(batch_x_time)], np.array(batch_y)
                batch_x_text, batch_x_user, batch_x_time, batch_y = [], [], [], []
        
        except Exception as e:
            print(f"Error processing chat {i}: {str(e)}")
        
        if i % 100 == 0:
            print(f"Processed {i}/{total_chats} chats...")
    
    if batch_x_text:
        yield [np.array(batch_x_text), np.array(batch_x_user), np.array(batch_x_time)], np.array(batch_y)
    
    print(f"Total unique topics: {len(topic_to_index)}")
    print(f"Topic to index mapping: {topic_to_index}")

def train_model(room_id, epochs=10):
    print("Starting enhanced model training...")
    num_classes = MAX_THREADS
    
    model = EnhancedDynamicClassifier(lstm_units=128, output_dim=EXPECTED_DIM, initial_num_classes=num_classes, dropout_rate=0.5)
    
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
    
    print("Enhanced model training completed")
    return model

def save_model(model):
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'enhanced_dynamic_classifier_model.keras')
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
model = EnhancedDynamicClassifier(lstm_units=128, output_dim=EXPECTED_DIM, initial_num_classes=INITIAL_NUM_CLASSES)

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
        
        original_topics = {str(chat['_id']): safe_int_convert(chat.get('topic', -1)) for chat in chat_collection.find({'room': ObjectId(room_id)})}
        
        predicted_topics = assign_topics(room_id)
        
        accuracy, report = evaluate_predictions(original_topics, predicted_topics)
        
        topic_distribution = Counter(predicted_topics.values())
        total_messages = len(predicted_topics)
        
        result = {
            'predicted_class': int(max(topic_distribution, key=topic_distribution.get)),
            'max_probability': float(max(topic_distribution.values()) / total_messages),
            'top_3_predictions': [
                {"class": int(cls), "probability": float(count / total_messages)}
                for cls, count in topic_distribution.most_common(3)
            ],
            'all_probabilities': [float(topic_distribution.get(i, 0) / total_messages) for i in range(1, MAX_THREADS + 1)],
            'message_count': int(total_messages),
            'topics': [int(t) for t in predicted_topics.values()],
            'accuracy': float(accuracy),
            'classification_report': report
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()   
        return jsonify({'error': str(e)}), 500

def normalize_database_topics(room_id):
    for chat in chat_collection.find({'room': ObjectId(room_id)}):
        original_topic = chat.get('topic')
        if original_topic is not None:
            normalized_topic = safe_int_convert(original_topic, default=-1)
            if normalized_topic != original_topic:
                chat_collection.update_one(
                    {'_id': chat['_id']},
                    {'$set': {'topic': normalized_topic}}
                )
                print(f"Updated topic for chat {chat['_id']}: {original_topic} -> {normalized_topic}")

if __name__ == '__main__':
    print("Starting enhanced CATD-COMBINE with new topic assignment...")
    room_id = '66b0fd658aab9f2bd7a41845'  # 여기에 실제 사용할 room_id를 입력하세요
    
    try:
        normalize_database_topics(room_id)
        model = train_model(room_id)
        save_model(model)
        
        port = int(os.environ.get('PORT', 5000))
        print(f"Server started on port {port}")
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")