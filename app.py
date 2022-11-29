import streamlit as st
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle as pkl
import sklearn

st.write("Check the vibe of your song!")
message_text = st.text_input("Enter the lyrics: ")

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

model = pkl.load(open("model.pkl", "rb"))

tokenizer = pkl.load(open('tokenizer.pkl', 'rb'))

x = []
sentences = list([message_text])
for sen in sentences:
    x.append(preprocess_text(sen))
x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, padding='post', maxlen=200)

labels = ['Depressed', 'Sad', 'Happy', 'Cheerful']

if message_text != '':
  st.write(labels[model.predict(x)[0]])