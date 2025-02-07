import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the model and tokenizer with error handling
try:
    model = BertForSequenceClassification.from_pretrained('trained_model')
    tokenizer = BertTokenizer.from_pretrained('trained_model')
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")

# Set the model to evaluation mode
model.eval()

# Define a function to make predictions
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()  # Get confidence score
    return "Positive" if predicted_class == 1 else "Negative", confidence

# Streamlit app layout
st.title("Sentiment Analysis with BERT")
st.write("Enter a movie review to analyze its sentiment:")

# Text input for user
user_input = st.text_area("Review:")

# Button to make prediction
if st.button("Analyze"):
    if user_input:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
    else:
        st.write("Please enter a review.")
