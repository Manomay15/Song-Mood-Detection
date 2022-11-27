import streamlit as st
import re
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle as pkl

st.write("Music mood classifier")
message_text = st.text_input("Enter the lyrics of your song: ")

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

model = load_model('new_model.h5')
tokenizer = pkl.load(open('tokenizer.pkl', 'rb'))

def tokenize(text):
   tokens = tokenizer.text_to_sequence(text)
   arr = pad_sequences(tokens, padding='post', maxlen=200)
   return arr

def classify_message(model, message):
  labels = model.predict(tokenize(message))
  value = labels.index(max(labels))
#   spam_prob = model.predict_proba([message])
  preds = ['Depressed', 'Sad', 'Happy', 'Cheerful']
  return preds[value]

if message_text != '':
  result = classify_message(model, message_text)
  st.write(result)