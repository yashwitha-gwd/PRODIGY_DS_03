import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from preprocessing import load_data, clean_data, encode_data

# Load data
df = load_data('../data/bank.csv')

# Clean data
df = clean_data(df)

# Split BEFORE encoding (important best practice)
X = df.drop('y', axis=1)
y = df['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Encode AFTER split (prevents leakage issues)
X_train = encode_data(X_train)
X_test = encode_data(X_test)

# Align columns (VERY IMPORTANT)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluation
print("Model Evaluation:")
print(classification_report(y_test, y_pred))

# Save model + columns
os.makedirs('../models', exist_ok=True)

pickle.dump(rf, open('../models/model.pkl', 'wb'))
pickle.dump(X_train.columns.tolist(), open('../models/columns.pkl', 'wb'))

print("Model and columns saved successfully!")