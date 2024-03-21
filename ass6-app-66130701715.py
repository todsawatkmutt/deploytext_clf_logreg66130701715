from sklearn.linear_model import LogisticRegression
import streamlit as st

# สร้างและฝึกโมเดล Logistic Regression
text_clf_logreg = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(max_iter=1000))
])

text_clf_logreg.fit(X_train, y_train)

# ประเมินประสิทธิภาพของโมเดล
logreg_train_accuracy = text_clf_logreg.score(X_train, y_train)
logreg_test_accuracy = text_clf_logreg.score(X_test, y_test)
print("Logistic Regression Training Accuracy:", logreg_train_accuracy)
print("Logistic Regression Test Accuracy:", logreg_test_accuracy)
