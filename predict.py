import os
import pickle
import pandas as pd
import numpy as np



from preprocessing import clean_data, encode_data

# ----------------------------
# Load model safely (FIXED PATH)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
columns_path = os.path.join(BASE_DIR, 'models', 'columns.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(columns_path, 'rb') as f:
    columns = pickle.load(f)

# ----------------------------
# Prediction function
# ----------------------------
def predict(input_df, threshold=0.5):
    if not isinstance(input_df, pd.DataFrame):
        raise ValueError("input_df must be a pandas DataFrame")

    df = input_df.copy()

    # 1. Clean data (IMPORTANT)
    df = clean_data(df)

    # 2. Encode using SAME logic as training
    df = encode_data(df)

    # 3. Align columns
    df = df.reindex(columns=columns, fill_value=0)

    # 4. Predict probabilities
    prob = model.predict_proba(df)[:, 1]

    # 5. Apply threshold
    pred = (prob >= threshold).astype(int)

    # 6. Convert to readable labels
    result = np.where(pred == 1, "Likely to Purchase", "Not Likely to Purchase")

    return result[0], prob[0]
