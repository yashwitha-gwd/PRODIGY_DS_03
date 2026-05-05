import pickle
import pandas as pd
import numpy as np

# Load model and columns
with open('../models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/columns.pkl', 'rb') as f:
    columns = pickle.load(f)

def predict(input_df, threshold=0.5):
    if not isinstance(input_df, pd.DataFrame):
        raise ValueError("input_df must be a pandas DataFrame")

    # 1. One-hot encoding
    input_df = pd.get_dummies(input_df)

    # 2. Align columns (very important)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # 3. Get probability safely
    prob = model.predict_proba(input_df)

    # always take probability of positive class
    if prob.shape[1] == 2:
        prob = prob[:, 1]
    else:
        prob = prob[:, 0]

    # 4. Apply threshold
    pred = (prob >= threshold).astype(int)

    # 5. Convert to readable labels
    result = np.where(pred == 1, "Likely to Purchase", "Not Likely to Purchase")

    return result, prob