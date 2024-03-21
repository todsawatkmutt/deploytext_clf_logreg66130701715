pip install scikit-learn
import pandas as pd
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Load your training data or define X_train, y_train, X_test, y_test here

# Define and train your model
text_clf_logreg = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(max_iter=1000))
])

text_clf_logreg.fit(X_train, y_train)

# Evaluate your model
logreg_train_accuracy = text_clf_logreg.score(X_train, y_train)
logreg_test_accuracy = text_clf_logreg.score(X_test, y_test)

# Display results using Streamlit
st.write("Logistic Regression Training Accuracy:", logreg_train_accuracy)
st.write("Logistic Regression Test Accuracy:", logreg_test_accuracy)
