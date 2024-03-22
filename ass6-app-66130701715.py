import pandas as pd
import pickle
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# URL ของไฟล์ CSV
csv_url = "https://github.com/todsawatkmutt/deploytext_clf_logreg66130701715/raw/main/review_shopping.csv"

# โหลดข้อมูล CSV จาก URL
df = pd.read_csv(csv_url)
# Load the trained model
model = pickle.load(open('text_clf_logreg-66130701715.sav', 'rb'))

# Split the text and label columns
df[['Review', 'Label']] = df.iloc[:, 0].str.split('\t', expand=True)

# Display a sample of the data
st.write(df.head())

# Evaluate the model
X_train = df['Review']
y_train = df['Label']
logreg_train_accuracy = model.score(X_train, y_train)
st.write("Logistic Regression Training Accuracy:", logreg_train_accuracy)

# Create a text input for testing the model
test_text = st.text_input("ใส่ข้อความทดสอบ:", "")

# Predict the label for the test text
if st.button("ทดสอบโมเดล"):
    prediction = model.predict([test_text])
    st.write("ผลการทดสอบ:", prediction)
