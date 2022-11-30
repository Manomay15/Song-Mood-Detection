import streamlit as st
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle as pkl
import sklearn
import xgboost

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

# model = pkl.load(open("xgboost.pkl", "rb"))
valence_model = pkl.load(open("xgvalence.pkl", "rb"))
tokenizer = pkl.load(open('tokenizer.pkl', 'rb'))

x = []
sentences = list([message_text])
for sen in sentences:
    x.append(preprocess_text(sen))
x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, padding='post', maxlen=500)

labels = ['depressing', 'sad', 'happy', 'cheerful']

valence = valence_model.predict(x)[0]

if valence <= 0.25:
    ind = 0
if valence > 0.25 and valence <= 0.5:
    ind = 1
if valence > 0.5 and valence <= 0.75:
    ind = 2
if valence > 0.75:
    ind = 3

if message_text != '':
  # st.write("The Song Appears To Be Pretty ", labels[model.predict(x)[0]])
  st.write("The spotify valence score is: ", valence_model.predict(x)[0], " which means the song is pretty ", labels[ind], "!")