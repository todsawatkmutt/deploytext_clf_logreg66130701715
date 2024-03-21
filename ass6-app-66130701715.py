from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# โหลดข้อมูล
df = pd.read_csv('review_shopping.csv', header=None)
df[['Review', 'Label']] = df[0].str.split('\t', expand=True)
df = df.drop(columns=[0])

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Label'], test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Naive Bayes
text_clf_nb = Pipeline([
    ('vect', CountVectorizer()),  # เปลี่ยนข้อความให้เป็นตัวเลข
    ('tfidf', TfidfTransformer()),  # ปรับน้ำหนักของคำ
    ('clf', MultinomialNB())  # ใช้โมเดล Naive Bayes
])

# ฝึกโมเดล
text_clf_nb.fit(X_train, y_train)

# ประเมินความแม่นยำของโมเดลบนชุดทดสอบ
accuracy = text_clf_nb.score(X_test, y_test)
print("Accuracy:", accuracy)
