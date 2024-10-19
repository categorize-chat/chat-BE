import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

app = Flask(__name__)

# MongoDB connection
mongodb_uri = os.getenv('MONGODB_URI')
client = MongoClient(mongodb_uri)
db = client.get_database()
chat_collection = db['chats']

# OpenAI client initialization
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
EMBEDDING_DIM = 1536
USER_EMB_DIM = 64
TIME_EMB_DIM = 64
ADDITIONAL_FEATURES = USER_EMB_DIM + TIME_EMB_DIM
EXPECTED_DIM = EMBEDDING_DIM + ADDITIONAL_FEATURES
MAX_THREADS = 15
MAX_THREAD_LENGTH = 20
BATCH_SIZE = 32

class CombinedCATDNSP(nn.Module):
    def __init__(self, lstm_units=256, attention_units=128, nsp_model_name="klue/bert-base", max_threads=MAX_THREADS):
        super(CombinedCATDNSP, self).__init__()
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.max_threads = max_threads

        # CATD components
        self.user_embedding_layer = nn.Embedding(1000, USER_EMB_DIM)
        self.time_embedding_layer = nn.Linear(1, TIME_EMB_DIM)
        self.thread_dense = nn.Linear(EXPECTED_DIM, self.lstm_units)
        self.thread_lstm = nn.LSTM(self.lstm_units, self.lstm_units, batch_first=True)
        self.message_dense = nn.Linear(EXPECTED_DIM, self.lstm_units)
        self.thread_attention_dense = nn.Linear(self.lstm_units, self.attention_units)
        self.message_attention_dense = nn.Linear(self.lstm_units, self.attention_units)
        self.time_attention_dense = nn.Linear(TIME_EMB_DIM, self.attention_units)
        self.attention = nn.MultiheadAttention(self.attention_units, 1)
        self.normalization = nn.LayerNorm(self.attention_units * 2)
        self.dense1 = nn.Linear(self.attention_units * 2, self.lstm_units)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.lstm_units, max_threads)

        # NSP components
        self.nsp_tokenizer = AutoTokenizer.from_pretrained(nsp_model_name)
        self.nsp_model = AutoModel.from_pretrained(nsp_model_name)

        # Combination layer
        self.combine_layer = nn.Linear(self.lstm_units + 768, max_threads)  # 768 is BERT's hidden size

    def forward(self, thread, message, user_ids, time_diffs, nsp_input_ids, nsp_attention_mask, nsp_token_type_ids):
        # CATD forward pass
        thread_processed = self.thread_dense(thread)
        user_emb_thread = self.user_embedding_layer(user_ids)
        time_emb_thread = self.time_embedding_layer(time_diffs.unsqueeze(-1))
        
        combined_thread = torch.cat([thread_processed, user_emb_thread, time_emb_thread], dim=-1)
        thread_output, _ = self.thread_lstm(combined_thread)
        
        message_output = self.message_dense(message)
        message_output = message_output.unsqueeze(1)
        
        thread_attention = self.thread_attention_dense(thread_output)
        message_attention = self.message_attention_dense(message_output)
        time_attention = self.time_attention_dense(time_emb_thread)
        
        context_vector, _ = self.attention(message_attention, thread_attention, thread_attention)
        time_context, _ = self.attention(message_attention, time_attention, time_attention)
        
        combined_context = torch.cat([context_vector, time_context], dim=-1)
        combined_context = combined_context.squeeze(1)
        
        normalized = self.normalization(combined_context)
        catd_output = self.dense1(normalized)
        catd_output = self.dropout(catd_output)

        # NSP forward pass
        nsp_outputs = self.nsp_model(input_ids=nsp_input_ids, 
                                     attention_mask=nsp_attention_mask, 
                                     token_type_ids=nsp_token_type_ids)
        nsp_output = nsp_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token output

        # Combine CATD and NSP outputs
        combined_output = torch.cat([catd_output, nsp_output], dim=-1)
        final_output = self.combine_layer(combined_output)

        return F.softmax(final_output, dim=-1)

    def expand_classifier(self, new_num_classes):
        old_weights, old_biases = self.classifier.weight.data, self.classifier.bias.data
        new_weights = F.pad(old_weights, (0, 0, 0, new_num_classes - self.classifier.out_features))
        new_biases = F.pad(old_biases, (0, new_num_classes - self.classifier.out_features))
        self.classifier = nn.Linear(self.lstm_units, new_num_classes)
        self.classifier.weight.data = new_weights
        self.classifier.bias.data = new_biases

        old_weights, old_biases = self.combine_layer.weight.data, self.combine_layer.bias.data
        new_weights = F.pad(old_weights, (0, 0, 0, new_num_classes - self.combine_layer.out_features))
        new_biases = F.pad(old_biases, (0, new_num_classes - self.combine_layer.out_features))
        self.combine_layer = nn.Linear(self.lstm_units + 768, new_num_classes)
        self.combine_layer.weight.data = new_weights
        self.combine_layer.bias.data = new_biases

def get_or_create_embedding(chat_id, content):
    chat_doc = chat_collection.find_one({"_id": chat_id}, {"embedding": 1})
    if chat_doc and 'embedding' in chat_doc:
        return torch.tensor(chat_doc['embedding'])
    else:
        print("Creating new embedding")
        try:
            response = openai_client.embeddings.create(model="text-embedding-3-small", input=content)
            embedding = response.data[0].embedding
            chat_collection.update_one({"_id": chat_id}, {"$set": {"embedding": embedding}}, upsert=True)
            return torch.tensor(embedding)
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            return torch.randn(EMBEDDING_DIM)  # Fallback to random embedding

def calc_time_diff(current_time, previous_time):
    time_diff = (current_time - previous_time).total_seconds() / 3600  # Convert to hours
    return torch.tensor(min(time_diff / 24, 1.0), dtype=torch.float32)  # Normalize to [0, 1] with max of 24 hours

def preprocess_input(thread, new_message, tokenizer, max_length=512):
    thread_embeds = []
    user_ids = []
    time_diffs = []

    for i, msg in enumerate(thread[-MAX_THREAD_LENGTH:]):
        embed = get_or_create_embedding(msg['_id'], msg['content'])
        thread_embeds.append(embed)
        user_ids.append(hash(msg['nickname']) % 1000)  # Simple hash function for user ID
        if i > 0:
            time_diffs.append(calc_time_diff(msg['createdAt'], thread[i-1]['createdAt']))
        else:
            time_diffs.append(torch.tensor(0.0))

    # Pad sequences if necessary
    while len(thread_embeds) < MAX_THREAD_LENGTH:
        thread_embeds.append(torch.zeros(EMBEDDING_DIM))
        user_ids.append(0)
        time_diffs.append(torch.tensor(0.0))

    message_embed = get_or_create_embedding(new_message['_id'], new_message['content'])

    # NSP preprocessing
    thread_text = " ".join([msg['content'] for msg in thread[-5:]])  # Last 5 messages
    nsp_inputs = tokenizer(thread_text, new_message['content'], 
                           return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    
    return {
        'thread': torch.stack(thread_embeds),
        'message': message_embed,
        'user_ids': torch.tensor(user_ids),
        'time_diffs': torch.stack(time_diffs),
        'nsp_input_ids': nsp_inputs['input_ids'],
        'nsp_attention_mask': nsp_inputs['attention_mask'],
        'nsp_token_type_ids': nsp_inputs['token_type_ids']
    }

class ChatDataset(Dataset):
    def __init__(self, room_id, tokenizer):
        self.chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.chats) - 1  # Exclude the first message

    def __getitem__(self, idx):
        thread = self.chats[:idx+1]
        new_message = self.chats[idx+1]
        inputs = preprocess_input(thread, new_message, self.tokenizer)
        label = torch.tensor(int(new_message.get('topic', 0)))  # Assuming 'topic' field exists
        return inputs, label

def save_model(model, path='combined_catd_nsp_model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'nsp_model_name': model.nsp_model.name_or_path,
    }, path)
    print(f"Model saved to {path}")

def load_model(path='combined_catd_nsp_model.pth'):
    if not os.path.exists(path):
        print(f"No saved model found at {path}")
        return None

    checkpoint = torch.load(path)
    model = CombinedCATDNSP(nsp_model_name=checkpoint['nsp_model_name'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    return model

def train_model(room_id, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedCATDNSP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = ChatDataset(room_id, model.nsp_tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

    save_model(model)
    return model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        room_id = data['room_id']
        
        chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1))
        result = []
        
        model = load_model()
        if model is None:
            return jsonify({'error': 'No trained model found'}), 404
        
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        for i, chat in enumerate(chats[1:]):  # Skip the first message
            thread = chats[:i+1]
            inputs = preprocess_input(thread[:-1], chat, model.nsp_tokenizer)
            
            with torch.no_grad():
                inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}  # Add batch dimension
                prediction = model(**inputs)
            predicted_topic = int(torch.argmax(prediction[0]))
            
            result.append({
                'content': chat['content'],
                'predicted_topic': predicted_topic
            })
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()   
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting combined CATD-NSP model...")
    room_id = '66b0fd658aab9f2bd7a41841'
    
    try:
        model = load_model()
        if model is None:
            print("Training new model...")
            model = train_model(room_id)
        
        port = int(os.environ.get('PORT', 5000))
        print(f"Server started on port {port}")
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")