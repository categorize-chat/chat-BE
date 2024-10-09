import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# 환경 변수 로드
load_dotenv()

# MongoDB 연결
mongodb_uri = os.getenv('MONGODB_URI')
client = MongoClient(mongodb_uri)
db = client.get_database()
chat_collection = db['chats']

def get_embedding(chat):
    # 채팅에서 임베딩 추출
    return np.array(chat.get('embedding', []))

def test_embeddings(room_id):
    # 특정 room_id에 해당하는 채팅 가져오기
    chats = list(chat_collection.find({'room': ObjectId(room_id)}).sort('createdAt', 1))
    
    # 임베딩 추출
    embeddings = [get_embedding(chat) for chat in chats]
    
    # 유효한 임베딩만 선택 (임베딩이 없거나 차원이 다른 채팅 제외)
    expected_dim = len(embeddings[0]) if embeddings else 0
    valid_indices = [i for i, emb in enumerate(embeddings) if emb.size == expected_dim]
    valid_embeddings = np.array([embeddings[i] for i in valid_indices])
    
    if valid_embeddings.size == 0:
        print("No valid embeddings found.")
        return
    
    print(f"Shape of valid_embeddings: {valid_embeddings.shape}")
    
    # DBSCAN 클러스터링
    dbscan = DBSCAN(eps=0.9, min_samples=5)
    clusters = dbscan.fit_predict(valid_embeddings)
    
    # t-SNE를 사용하여 고차원 임베딩을 2D로 축소
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(valid_embeddings)
    
    # 시각화
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Chat Embeddings Clustering')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    
    # 각 클러스터에 레이블 추가
    for i in range(len(embeddings_2d)):
        plt.annotate(clusters[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.show()
    
    # 클러스터별 채팅 내용 출력
    for cluster in set(clusters):
        print(f"\nCluster {cluster}:")
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster]
        for idx in cluster_indices[:5]:  # 각 클러스터의 처음 5개 메시지만 출력
            original_idx = valid_indices[idx]
            print(f"- {chats[original_idx]['content'][:50]}...")

if __name__ == "__main__":
    room_id = '66b0fd658aab9f2bd7a41845'  # 테스트할 room_id
    test_embeddings(room_id)