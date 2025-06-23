import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# TensorFlow / Keras for TextCNN and Hybrid Model
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
tf.config.set_visible_devices([], 'GPU')
try:
    df = pd.read_csv('./Web_Datasets.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Web_Datasets.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# --- 2. Data Preparation ---
df = df.drop('Fid', axis=1)
X = df.drop('Label', axis=1)
y = df['Label']
numerical_features = ['Sender ID', 'Receiver ID', 'Source Port', 'Destination Port', 'Packet Size']
categorical_features = ['Protocol', 'Flag', 'Packet', 'Source IP Address', 'Destination IP Address']
text_feature = 'Url'

# Create and fit the preprocessor for structured data (numerical and categorical)
structured_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# Split data before fitting preprocessors to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit and transform the structured data
X_train_structured = structured_preprocessor.fit_transform(X_train)
X_test_structured = structured_preprocessor.transform(X_test)

# Convert sparse matrices to dense arrays
X_train_structured_dense = X_train_structured.toarray() if hasattr(X_train_structured, 'toarray') else X_train_structured
X_test_structured_dense = X_test_structured.toarray() if hasattr(X_test_structured, 'toarray') else X_test_structured

# --- NLP Preprocessing for URLs (TextCNN) ---
def preprocess_url(url):
    if not isinstance(url, str):
        return ""
    url = re.sub(r'^(http|https|ftp)://', '', url)
    url = re.sub(r'[^\w\s/.-]', ' ', url)
    url = url.replace('/', ' ').replace('.', ' ').replace('-', ' ')
    return url.lower()

# Apply preprocessing to URL column
X_train_urls = X_train[text_feature].apply(preprocess_url)
X_test_urls = X_test[text_feature].apply(preprocess_url)

# Tokenize and pad the URL text
vocab_size = 2000  # Max number of words to keep
max_length = 50    # Max length of a URL sequence
embedding_dim = 32 # Dimension of the word embedding

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<oov>')
tokenizer.fit_on_texts(X_train_urls)

X_train_sequences = tokenizer.texts_to_sequences(X_train_urls)
X_test_sequences = tokenizer.texts_to_sequences(X_test_urls)

X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post', truncating='post')

# --- 4. Building the Hybrid TextCNN Model ---

# Input layer for structured data (numerical + categorical)
input_structured = Input(shape=(X_train_structured_dense.shape[1],), name='structured_input')
# Input layer for text data (padded URL sequences)
input_text = Input(shape=(max_length,), name='text_input')

# --- TextCNN Branch ---
text_branch = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(input_text)
text_branch = Conv1D(filters=64, kernel_size=3, activation='relu')(text_branch)
text_branch = GlobalMaxPooling1D()(text_branch)
text_branch = Dense(32, activation='relu')(text_branch)
text_branch_model = Model(inputs=input_text, outputs=text_branch)

# --- Structured Data Branch ---
structured_branch = Dense(64, activation='relu')(input_structured)
structured_branch = Dropout(0.3)(structured_branch)
structured_branch = Dense(32, activation='relu')(structured_branch)
structured_branch_model = Model(inputs=input_structured, outputs=structured_branch)

# --- Concatenate Branches ---
combined = Concatenate()([structured_branch_model.output, text_branch_model.output])

# --- Final Classifier Head ---
final_layers = Dense(64, activation='relu')(combined)
final_layers = Dropout(0.5)(final_layers)
output = Dense(1, activation='sigmoid')(final_layers)  # Sigmoid for binary classification

# --- Create and Compile the Final Model ---
model = Model(inputs=[input_structured, input_text], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
print("\n--- Model Architecture ---")
model.summary()

# --- 5. Training the Model ---
print("\n--- Training the Hybrid Model ---")
history = model.fit(
    [X_train_structured_dense, X_train_padded],
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# --- 6. Evaluating the Model ---
print("\n--- Evaluating the Model on Test Data ---")
loss, accuracy = model.evaluate([X_test_structured_dense, X_test_padded], y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# Generate predictions and classification report
y_pred_prob = model.predict([X_test_structured_dense, X_test_padded])
y_pred = (y_pred_prob > 0.5).astype(int)

report = classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malicious (1)'])
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(report)

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign (0)', 'Malicious (1)'],
            yticklabels=['Benign (0)', 'Malicious (1)'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Hybrid TextCNN Model')
plt.show()

# --- Save the Model, Tokenizer, and Preprocessor ---
model.save('hybrid_model.h5')
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(structured_preprocessor, f)

# --- 7. Interactive Prediction ---
print("\n--- Interactive Prediction Mode ---")
print("Enter the details for a network session to classify it as Benign (0) or Malicious (1).")
print("Type 'exit' to quit.")

def predict_live():
    while True:
        try:
            # Collect user input
            input_data = {}
            for feature in numerical_features + categorical_features:
                val = input(f"Enter {feature}: ")
                if val.lower() == 'exit':
                    return
                try:
                    # Convert numerical features to appropriate type
                    if feature in numerical_features:
                        input_data[feature] = [float(val)]
                    else:
                        input_data[feature] = [val]
                except ValueError:
                    print(f"Invalid input for {feature}. Please enter a valid value.")
                    return

            # Get the URL separately
            url_val = input(f"Enter {text_feature}: ")
            if url_val.lower() == 'exit':
                return
            input_data[text_feature] = [url_val]

            # Create a DataFrame from the input
            input_df = pd.DataFrame(input_data)

            # --- Preprocess the live data ---
            # 1. Structured data
            input_structured_data = structured_preprocessor.transform(input_df)
            input_structured_data = input_structured_data.toarray() if hasattr(input_structured_data, 'toarray') else input_structured_data

            # 2. URL data
            input_url = input_df[text_feature].apply(preprocess_url)
            input_sequence = tokenizer.texts_to_sequences(input_url)
            input_padded = pad_sequences(input_sequence, maxlen=max_length, padding='post', truncating='post')

            # --- Make Prediction ---
            prediction_prob = model.predict([input_structured_data, input_padded], verbose=0)[0][0]
            prediction = (prediction_prob > 0.5).astype(int)
            result = "Malicious (1)" if prediction == 1 else "Benign (0)"

            print("\n---------------------------------")
            print(f"Prediction: {result}")
            print(f"Confidence (Malicious Probability): {prediction_prob:.4f}")
            print("---------------------------------\n")

        except Exception as e:
            print(f"\nAn error occurred: {e}. Please try again.\n")

# Start the interactive prediction loop
predict_live()