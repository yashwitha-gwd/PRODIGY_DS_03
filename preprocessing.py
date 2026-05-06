import pandas as pd

def load_data(path):
    df = pd.read_csv("bank.csv", sep=';', encoding='latin-1')
    return df

def clean_data(df):
    for col in df.columns:
        df[col] = df[col].replace('unknown', df[col].mode()[0])
    return df

def encode_data(df):
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    if 'duration' in df.columns:
        df = df.drop('duration', axis=1)
    
    df = pd.get_dummies(df, drop_first=True)
    return df

def split_data(df):
    X = df.drop('y', axis=1)
    y = df['y']
    return X, y
