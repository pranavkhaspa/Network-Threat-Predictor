
# 🌐 Network Threat Predictor

A **FastAPI-powered web application** that leverages a hybrid deep learning model to classify network sessions as **malicious** or **benign** using structured network traffic data and URL content.

🔗 **Live Demo**: [https://network-traffic-predictor.onrender.com](https://network-traffic-predictor.onrender.com)

---

## 📂 Project Structure

```
Network-Traffic-Predictor/
├── server.py              # FastAPI backend with prediction endpoint
├── train_model.py         # Script to train and save the hybrid model
├── Web_Datasets.csv       # Dataset for model training
├── hybrid_model.h5        # Trained hybrid model (structured + text input)
├── preprocessor.pkl       # Preprocessor for structured features
├── tokenizer.pkl          # Tokenizer for URL text processing
├── index.html             # Landing page
├── predict.html           # Prediction interface
├── static/                # Static files (CSS/JS)
├── requirements.txt       # Python dependencies
├── instructions.txt       # Setup and usage instructions
└── myenv/                 # Python virtual environment (optional)
```

---

## 🚀 Features

- **Hybrid Deep Learning Model**: Combines structured data (e.g., IP addresses, ports) with unstructured URL text for accurate predictions.
- **FastAPI Backend**: Lightweight, high-performance API for real-time predictions.
- **User-Friendly Web UI**: Accessible via `index.html` (homepage) and `predict.html` (prediction form).
- **REST API**: Programmatically submit network session data via POST to `/predict`.

---

## 🔧 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Network-Traffic-Predictor.git
   cd Network-Traffic-Predictor
   ```

2. **Create a Virtual Environment** (optional but recommended)
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Server**
   ```bash
   python server.py
   ```
   The server will be available at: `http://127.0.0.1:8000`

---

## 📡 Usage

### Web Interface
- Visit `http://127.0.0.1:8000` for the homepage.
- Navigate to `/predict` to access the prediction form.

### API Request
Send a POST request to `/predict` with the following JSON payload:
```json
{
  "sender_id": 101,
  "receiver_id": 202,
  "source_port": 443,
  "destination_port": 52345,
  "packet_size": 1500,
  "protocol": "TCP",
  "flag": "SYN",
  "packet": "ACK",
  "source_ip_address": "192.168.1.2",
  "destination_ip_address": "10.0.0.5",
  "url": "http://example.com/malicious-path"
}
```

**Sample Response**:
```json
{
  "prediction": "Malicious (1)",
  "confidence": 0.8732
}
```

---

## 🧠 How It Works

1. **Structured Data**: Preprocessed using `preprocessor.pkl` to handle numerical and categorical features.
2. **URL Text**: Cleaned, tokenized, and padded using `tokenizer.pkl`.
3. **Hybrid Model**: Combines both inputs to output a binary classification (malicious or benign).

---

## 📦 Dependencies

The project uses the following dependencies (specified in `requirements.txt`):

```
fastapi==0.111.0
uvicorn==0.30.0
tensorflow==2.15.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
```

Additional dependencies (e.g., `pickle`) are included in the standard library or as sub-dependencies.

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to fork, modify, and contribute!

---

## 🙋‍♂️ Author

**Pranav Khaspa**

Connect with me or contribute to the project on [GitHub](https://github.com/pranavkhaspa). Feedback and improvements are welcome!
