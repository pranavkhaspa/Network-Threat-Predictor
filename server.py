# server.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import pickle
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and preprocessor
model = tf.keras.models.load_model('hybrid_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    structured_preprocessor = pickle.load(f)

# FastAPI app
app = FastAPI(title="Network Threat Predictor")

# Serve static files (if you have a /static folder — optional)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Features
numerical_features = ['Sender ID', 'Receiver ID', 'Source Port', 'Destination Port', 'Packet Size']
categorical_features = ['Protocol', 'Flag', 'Packet', 'Source IP Address', 'Destination IP Address']
text_feature = 'Url'
max_length = 50

# URL Preprocessing
def preprocess_url(url: str) -> str:
    url = re.sub(r'^(http|https|ftp)://', '', url)
    url = re.sub(r'[^\w\s/.-]', ' ', url)
    url = url.replace('/', ' ').replace('.', ' ').replace('-', ' ')
    return url.lower()

# Pydantic input model
class NetworkSession(BaseModel):
    sender_id: float
    receiver_id: float
    source_port: float
    destination_port: float
    packet_size: float
    protocol: str
    flag: str
    packet: str
    source_ip_address: str
    destination_ip_address: str
    url: str

# Serve landing page
@app.get("/", response_class=HTMLResponse)
async def serve_landing():
    return FileResponse("index.html")

# Serve predictor page
@app.get("/predict", response_class=HTMLResponse)
async def serve_predict():
    return FileResponse("predict.html")

# API: POST /predict
@app.post("/predict")
async def predict(session: NetworkSession):
    try:
        # Build DataFrame
        input_data = {
            'Sender ID': [session.sender_id],
            'Receiver ID': [session.receiver_id],
            'Source Port': [session.source_port],
            'Destination Port': [session.destination_port],
            'Packet Size': [session.packet_size],
            'Protocol': [session.protocol],
            'Flag': [session.flag],
            'Packet': [session.packet],
            'Source IP Address': [session.source_ip_address],
            'Destination IP Address': [session.destination_ip_address],
            'Url': [session.url]
        }

        input_df = pd.DataFrame(input_data)

        # Preprocess structured data
        structured_data = structured_preprocessor.transform(input_df)
        structured_data = structured_data.toarray() if hasattr(structured_data, 'toarray') else structured_data

        # Preprocess URL
        url_processed = preprocess_url(session.url)
        url_sequence = tokenizer.texts_to_sequences([url_processed])
        url_padded = pad_sequences(url_sequence, maxlen=max_length, padding='post', truncating='post')

        # Predict
        prediction_prob = model.predict([structured_data, url_padded], verbose=0)[0][0]
        prediction = int(prediction_prob > 0.5)
        label = "Malicious (1)" if prediction == 1 else "Benign (0)"

        return {
            "prediction": label,
            "confidence": round(float(prediction_prob), 4)
        }

    except Exception as e:
        return {"error": str(e)}

# Run app
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
