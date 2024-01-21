import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

def transform_text(text):
    text=text.lower()    #converting the text into lowercase
    text=nltk.word_tokenize(text)    #using nltk for tokenization and gives each word of the text in the form of list
    # As the text is now converted in the form of list,now we will run the loop and then alphanumeric only taken
    y=[]    #initialize empty list
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    for i in text:
            y.append(ps.stem(i))
    
    
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl', 'rb'))
model=pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')
input_sms=st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms=transform_text(input_sms)
    vector_input=tfidf.transform([transformed_sms])
    result=model.predict(vector_input)[0]
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')

#STEPS
#1) Preprocess the input_sms
#2) Vectorize the input_sms
#3)predict
#4) Display


