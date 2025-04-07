import os
import torch
from transformers import AutoTokenizer, AutoModelForNextSentencePrediction
from sentence_transformers import SentenceTransformer

# 모델 캐시 경로 설정
os.environ['TRANSFORMERS_CACHE'] = '/app/model_cache'
os.environ['HF_HOME'] = '/app/model_cache'

# 캐시 디렉토리 생성
os.makedirs('/app/model_cache', exist_ok=True)

# 장치 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# KLUE BERT 모델 및 토크나이저 다운로드
print("Downloading KLUE BERT model and tokenizer...")
klue_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
klue_model = AutoModelForNextSentencePrediction.from_pretrained("klue/bert-base")

# KR-SBERT 모델 다운로드
print("Downloading KR-SBERT model...")
sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', cache_folder='/app/model_cache')

print("Model download completed. Models saved to /app/model_cache") 