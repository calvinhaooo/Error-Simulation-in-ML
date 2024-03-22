import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype, is_datetime64_ns_dtype
from pandas import DataFrame
import numpy as np

def load_datasets(electronics_path, modcloth_path):
    electronics = pd.read_csv(electronics_path)
    modcloth = pd.read_csv(modcloth_path)
    return electronics, modcloth

def preprocess_electronics(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['timestamp'] = data['timestamp'].dt.tz_localize('UTC')
    data['user_id'] = data['user_id'].astype(str)
    return data

def preprocess_modcloth(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['size'] = data['size'].fillna(data['size'].median())
    return data

def merge_data(modcloth, electronics, on_column):
    
    merged_df = pd.merge(modcloth, electronics, on=on_column, how='outer')

    return merged_df

def select_features(data, label, alpha=0.1, beta=1e-3):
    categorical_columns, numerical_columns, text_columns, timestamps = [], [], [], []
    for column in data.columns:
        if column == label:
            continue

        null_num = data[column].isnull().sum()
        missing_percent = null_num / len(data)
        if missing_percent > alpha:
            missing_percentage = "{0:.2f}".format(missing_percent * 100)
            print(f"Drop column {column} for {missing_percentage}% missing values")
            continue

        data_type = data[column].dtype
        feature_num = data[column].nunique()
        if is_numeric_dtype(data_type):
            numerical_columns.append(column)
        elif feature_num <= len(data) * beta:
            categorical_columns.append(column)
        elif is_string_dtype(data_type):
            text_columns.append(column)
        elif is_datetime64_ns_dtype(data_type):
            timestamps.append(column)

    return categorical_columns, numerical_columns, text_columns, timestamps

def merge_text(data: DataFrame, text_columns, column_name):
    data[column_name] = ''
    for text_column in text_columns:
        data[column_name] += ' ' + data[text_column]