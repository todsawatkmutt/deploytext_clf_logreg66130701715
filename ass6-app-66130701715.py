pip install -r requirements1.txt
import pandas as pd
import pickle
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Load the data
df = pd.read_csv('review_shopping.csv', header=None)

# Split the text and label columns
df[['Review', 'Label']] = df[0].str.split('\t', expand=True)
df = df.drop(columns=[0])

# Display a sample of the data
st.write(df.head())

# Create and train the Logistic Regression model
text_clf_logreg = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(max_iter=1000))
])

X_train = df['Review']
y_train = df['Label']

text_clf_logreg.fit(X_train, y_train)

# Evaluate the model
logreg_train_accuracy = text_clf_logreg.score(X_train, y_train)
st.write("Logistic Regression Training Accuracy:", logreg_train_accuracy)
# Create a text input for testing the model
test_text = st.text_input("ใส่ข้อความทดสอบ:", "")

# Predict the label for the test text
if st.button("ทดสอบโมเดล"):
    prediction = text_clf_logreg.predict([test_text])
    st.write("ผลการทดสอบ:", prediction)
