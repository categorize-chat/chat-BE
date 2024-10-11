import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoModel, DistilBertForMaskedLM, TFBertForNextSentencePrediction

model = TFBertForNextSentencePrediction.from_pretrained('klue/bert-base', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

def NSP(base_message, target_message):
    encoding = tokenizer(base_message, target_message, return_tensors='tf') 

    logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]

    softmax = tf.keras.layers.Softmax()
    probs = softmax(logits)

    is_same_class = not bool(tf.math.argmax(probs, axis=-1).numpy())
    prob = np.max(probs, axis=-1)[0]

    return is_same_class, prob

ans, prob = NSP('점심 뭐 먹을래요? 탕수육이 괜찮을 것 같아요. 마라탕은 어때요?', '파란색 고양이가 있어요')

print(ans, prob)