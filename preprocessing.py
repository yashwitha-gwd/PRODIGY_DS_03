import pandas as pd


def load_data(path):
    df = pd.read_csv(path, sep=';', encoding='latin-1')
    return df


def clean_data(df):
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace('unknown', df[col].mode()[0])
    
    return df


def encode_data(df):
    df = df.copy()
    
    # Drop leakage column
    if 'duration' in df.columns:
        df = df.drop('duration', axis=1)
    
    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    
    return df


def split_data(df):
    X = df.drop('y', axis=1)
    y = df['y']
    return X, y
