import joblib
import os
import re
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from underthesea import word_tokenize
import tensorflow as tf
# uvicorn main:app --reload
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "models"

def load_model(path):
    try:
        if path.endswith('.keras'):
            return tf.keras.models.load_model(path)
        return joblib.load(path)
    except Exception as e:
        print(f"Lỗi khi tải file: {path}. Lỗi: {e}")
        return None

# TF-IDF Models
tfidf_vectorizer = load_model(os.path.join(MODEL_DIR, 'TF-IDF/tfidf_vectorizer.joblib'))
model_svm_tfidf = load_model(os.path.join(MODEL_DIR, 'TF-IDF/model_knn.joblib'))
model_lr_tfidf = load_model(os.path.join(MODEL_DIR, 'TF-IDF/model_logistic_regression.joblib'))
model_knn_tfidf = load_model(os.path.join(MODEL_DIR, 'TF-IDF/model_knn.joblib'))
model_nb_tfidf = load_model(os.path.join(MODEL_DIR, 'TF-IDF/model_naive_bayes.joblib'))
model_rf_tfidf = load_model(os.path.join(MODEL_DIR, 'TF-IDF/model_random_forest.joblib'))

# Bag-of-Words Models
bow_vectorizer = load_model(os.path.join(MODEL_DIR, 'BoW/bow_vectorizer.joblib'))
model_mlp_bow = load_model(os.path.join(MODEL_DIR, 'BoW/bow_model_mlp.keras'))
model_lr_bow = load_model(os.path.join(MODEL_DIR, 'BoW/bow_model_logistic_regression.joblib'))
model_knn_bow = load_model(os.path.join(MODEL_DIR, 'BoW/bow_model_knn.joblib'))
model_nb_bow = load_model(os.path.join(MODEL_DIR, 'BoW/bow_model_naive_bayes.joblib'))
model_rf_bow = load_model(os.path.join(MODEL_DIR, 'BoW/bow_model_random_forest.joblib'))
model_svm_bow = load_model(os.path.join(MODEL_DIR, 'BoW/bow_model_svm.joblib'))

# RNN Models
rnn_config = load_model(os.path.join(MODEL_DIR, 'TF-IDF/rnn_tokenizer_config.joblib'))
rnn_tokenizer = rnn_config['tokenizer'] if rnn_config else None
MAX_LEN_RNN = rnn_config['max_len'] if rnn_config else 256
model_rnn = load_model(os.path.join(MODEL_DIR, 'TF-IDF/model_rnn.keras'))

label_encoder = load_model(os.path.join(MODEL_DIR, 'label_encoder.joblib'))


vietnamese_stopwords = [
    'và', 'của', 'là', 'cho', 'có', 'không', 'được', 'để', 'với', 'trong', 'trên', 'dưới', 'để', 'của', 'với', 'từ',
    'có', 'là', 'được', 'sẽ', 'đã', 'đang', 'các', 'một', 'hai', 'ba', 'này', 'đó',
    'khi', 'nếu', 'vì', 'do', 'theo', 'như', 'về', 'đến', 'cho', 'bởi', 'tại', 'sau',
    'trước', 'giữa', 'cùng', 'mà', 'rằng', 'ở', 'ra', 'vào', 'lại', 'đi', 'lên', 'xuống',
    'thì', 'mới', 'đều', 'cả', 'tất', 'toàn', 'nào', 'ai', 'gì', 'đâu', 'sao', 'bao',
    'bằng', 'rất', 'quá', 'khá', 'hơn', 'nữa', 'thêm', 'chỉ', 'chính', 'cũng', 'còn',
    'lại', 'phải', 'nên', 'cần', 'muốn', 'không', 'chẳng', 'chưa', 'đừng', 'hãy',
    'bài', 'viết', 'genk', 'đầy', 'đủ', 'thể', 'rất'
]

def preprocess_text(text: str):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]', ' ', text)
    tokens = word_tokenize(text, format="text")
    filtered_tokens = [word for word in tokens.split() if word not in vietnamese_stopwords]
    return " ".join(filtered_tokens)

class PredictRequest(BaseModel):
    text: str
    feature_type: str
    model_name: str

class PredictResponse(BaseModel):
    category: str
    confidence: float = 0.0

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    processed_text = preprocess_text(request.text)
    
    prediction_idx = -1
    confidence = 0.0
    
    models_dict = {
        'tfidf': {
            'svm': model_svm_tfidf, 'lr': model_lr_tfidf, 'knn': model_knn_tfidf,
            'nb': model_nb_tfidf, 'rf': model_rf_tfidf
        },
        'bow': {
            'svm': model_svm_bow, 'lr': model_lr_bow, 'knn': model_knn_bow,
            'nb': model_nb_bow, 'rf': model_rf_bow, 'mlp': model_mlp_bow
        },
        'rnn': {
            'rnn_lstm': model_rnn
        }
    }
    
    model_to_use = models_dict.get(request.feature_type, {}).get(request.model_name)

    if not model_to_use:
        return PredictResponse(category="Mô hình không hợp lệ", confidence=0.0)

    if request.feature_type == 'tfidf':
        vectorized_text = tfidf_vectorizer.transform([processed_text])
        prediction_idx = model_to_use.predict(vectorized_text)[0]
        if hasattr(model_to_use, 'predict_proba'):
            confidence = np.max(model_to_use.predict_proba(vectorized_text))

    elif request.feature_type == 'bow':
        vectorized_text = bow_vectorizer.transform([processed_text])
        if request.model_name == 'mlp':
            pred_probs = model_to_use.predict(vectorized_text.toarray())
            prediction_idx = np.argmax(pred_probs, axis=1)[0]
            confidence = np.max(pred_probs)
        else:
            prediction_idx = model_to_use.predict(vectorized_text)[0]
            if hasattr(model_to_use, 'predict_proba'):
                confidence = np.max(model_to_use.predict_proba(vectorized_text))

    elif request.feature_type == 'rnn':
        seq = rnn_tokenizer.texts_to_sequences([processed_text])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN_RNN, padding='post', truncating='post')
        pred_probs = model_to_use.predict(padded_seq)
        prediction_idx = np.argmax(pred_probs, axis=1)[0]
        confidence = np.max(pred_probs)

    if prediction_idx != -1 and label_encoder:
        category = label_encoder.inverse_transform([prediction_idx])[0]
        return PredictResponse(category=category, confidence=float(confidence))

    return PredictResponse(category="Không xác định", confidence=0.0)

@app.get("/")
def read_root():
    return {"message": "News Classification API is running"}
