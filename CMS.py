import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv(r'csv file path', low_memory=False)

# Ensure 'comment_text' is a string and handle NaN values
df['comment_text'] = df['comment_text'].fillna("").astype(str)

df['label'] = df['toxic_score'].apply(lambda x: 1 if x > 0.6 else 0)

df['comment_length'] = df['comment_text'].apply(lambda x: len(x.split()))

df = df.sample(frac=1, random_state=42)  

# Split data
X = df[['comment_text', 'comment_length']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train['comment_text'])
X_test_tfidf = tfidf.transform(X_test['comment_text'])

# Combine Tfidf features and comment length
import scipy
X_train_combined = scipy.sparse.hstack((X_train_tfidf, X_train[['comment_length']].values))
X_test_combined = scipy.sparse.hstack((X_test_tfidf, X_test[['comment_length']].values))

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_combined, y_train)

# Logistic Regression with RandomizedSearch
param_dist = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}
clf = RandomizedSearchCV(
    LogisticRegression(max_iter=2000, class_weight='balanced'), 
    param_dist, 
    n_iter=5,  # Try fewer combinations
    cv=3,      # 3-fold cross-validation
    scoring='accuracy', 
    n_jobs=-1
)
clf.fit(X_train_res, y_train_res)

# Evaluate the model
y_pred = clf.best_estimator_.predict(X_test_combined)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

