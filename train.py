import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from preprocessing import load_data, clean_data, encode_data


df = load_data('../data/bank.csv')


df = clean_data(df)


df['y'] = df['y'].astype(str).str.strip().str.lower()
df = df[df['y'].isin(['yes', 'no'])]

y = df['y'].map({'yes': 1, 'no': 0})
X = df.drop('y', axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = encode_data(X_train)
X_test = encode_data(X_test)

# Align columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)

print("Model Evaluation:")
print(classification_report(y_test, y_pred))


os.makedirs('../models', exist_ok=True)

with open('../models/model.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('../models/columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

print("✅ Model and columns saved successfully!")
