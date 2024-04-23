import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

file_name = "Spam_sms_prediction.pkl"
classifier = pickle.load(open(file_name, 'rb'))

file_name = "corpus.pkl"
corpus = pickle.load(open(file_name, 'rb'))

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

def predict_spam(sample_message):
    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return classifier.predict(temp)

st.header("SPAM message Detector")
name = st.text_input("Enter message")
ans = name.title()
press=st.button("Submit")
if(press):
    score=predict_spam(ans)
    if score==1:
        st.image('spam.png',width=500)
    else:
        st.image("nospam.png",width=300)
