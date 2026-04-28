import pandas as pd
import numpy as np
from datetime import datetime
import holidays

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers(df, threshold):
    return df[df['sales'] < threshold].copy()

def extract_features(df):
    df = df.copy()
    
    parts = df["date"].str.split("-", n=3, expand=True)
    df["year"] = parts[0].astype('int')
    df["month"] = parts[1].astype('int')
    df["day"] = parts[2].astype('int')

    df['weekday'] = df.apply(lambda x: datetime(x['year'], x['month'], x['day']).weekday(), axis=1)
    df['weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    india_holidays = holidays.country_holidays('IN') # India hahahahaha
    df['holidays'] = df['date'].apply(lambda x: 1 if india_holidays.get(x) else 0)

    df['m1'] = np.sin(df['month'] * (2 * np.pi / 12))
    df['m2'] = np.cos(df['month'] * (2 * np.pi / 12))

    df.drop(['date', 'year'], axis=1, inplace=True)
    
    return df

def prepare_pipeline(filepath, outlier_threshold):
    """Orquestra as etapas de preparação."""
    df = load_data(filepath)
    df = remove_outliers(df, outlier_threshold)
    df = extract_features(df)
    return df