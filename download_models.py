import os
import torch
import time
import sys
from transformers import AutoTokenizer, AutoModelForNextSentencePrediction
from sentence_transformers import SentenceTransformer

def download_models():
    try:
        # 모델 캐시 경로 설정
        os.environ['TRANSFORMERS_CACHE'] = '/app/model_cache'
        os.environ['HF_HOME'] = '/app/model_cache'
        os.environ['TORCH_HOME'] = '/app/model_cache'

        # 캐시 디렉토리 생성
        os.makedirs('/app/model_cache', exist_ok=True)

        # 장치 확인
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # KLUE BERT 모델 및 토크나이저 다운로드
        print("Downloading KLUE BERT model and tokenizer...")
        klue_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", cache_dir='/app/model_cache')
        klue_model = AutoModelForNextSentencePrediction.from_pretrained("klue/bert-base", cache_dir='/app/model_cache')
        print("KLUE BERT model and tokenizer downloaded successfully.")

        # KR-SBERT 모델 다운로드
        print("Downloading KR-SBERT model...")
        sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', cache_folder='/app/model_cache')
        print("KR-SBERT model downloaded successfully.")

        # 모델 파일 확인
        model_files = os.listdir('/app/model_cache')
        print(f"Model cache contents ({len(model_files)} items):")
        
        # 파일 크기 계산
        total_size = 0
        for root, dirs, files in os.walk('/app/model_cache'):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        
        print(f"Total model size: {total_size / (1024*1024*1024):.2f} GB")
        print("Model download completed. Models saved to /app/model_cache")
        
        return True
    except Exception as e:
        print(f"Error downloading models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting model download process...")
    
    # 최대 3번 재시도
    max_retries = 3
    for attempt in range(max_retries):
        if download_models():
            sys.exit(0)
        else:
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"Download failed. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print("All download attempts failed. Please check network and try again.")
                sys.exit(1) 