import random
import string

import numpy as np
import pandas as pd
from pandas import DataFrame


def add_noise(df, column, mean=0):
    mc_data = df.copy(deep=True)
    stddev = np.std(mc_data[column])
    sampled_rows = mc_data.sample(frac=0.3)
    noise = np.random.normal(mean, stddev, size=len(sampled_rows))
    df.loc[sampled_rows.index, column] += noise


def duplicates_data(data: DataFrame, percentage=3):
    total_number = int(len(data) * percentage / 100)
    random_rows = data.sample(n=total_number)
    random_rows = DataFrame(random_rows)
    new_data = pd.concat([data, random_rows])
    return new_data


def add_noise_to_text(df: DataFrame, percentage=10, noise_percentage=10):
    num_outliers = int(len(df) * percentage / 100)
    indices = np.random.choice(df.index, num_outliers, replace=False)
    characters = string.punctuation + string.digits
    for i in indices:
        text = df.loc[i, 'text']
        num_noise_chars = int(len(text) * noise_percentage / 100)

        for _ in range(num_noise_chars):
            index = random.randint(0, len(text) - 1)
            noise_char = random.choice(characters)
            text = text[:index] + noise_char + text[index:]

        df.loc[i, 'text'] = text


def add_null_noise(df, label_column, null_percentage=5):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_modify = [col for col in numeric_cols if col != label_column]
    num_nulls = int(len(df) * null_percentage / 100)
    null_indices = np.random.choice(df.index, num_nulls, replace=False)
    for index in null_indices:
        col = np.random.choice(cols_to_modify)
        df.at[index, col] = np.nan


def random_replace_column(df, column, num_labels):
    print(df[column].dtype)
    if df[column].dtype == 'object':
        # num_labels = int(len(df) * outlier_percentage / 100)
        label_indices = np.random.choice(df.index, num_labels, replace=False)
        for index in label_indices:
            label = df.at[index, column]
            char_list = list(label)

            replace_index = random.randint(0, len(char_list) - 1)
            new_char = random.choice('a')

            char_list[replace_index] = new_char
            result_str = ''.join(char_list)

            df.at[index, column] = result_str
    else:
        label_indices = np.random.choice(df.index, num_labels, replace=False)
        outliers = np.random.normal(np.mean(df[column]) * 2, np.std(df[column]) * 3, num_labels)
        df.loc[label_indices, column] += outliers


def add_outliers(df, numerical_columns, categorical_columns, outlier_percentage=5, factor=3):
    num_outliers = int(len(df) * outlier_percentage / 100)
    print(num_outliers)
    outlier_indices = np.random.choice(df.index, num_outliers, replace=False)
    # outliers = np.random.normal(np.mean(df[column]) * factor, np.std(df[column]) * factor, num_outliers)
    # outliers = df.loc[outlier_indices, column]
    df.loc[outlier_indices, numerical_columns] *= factor
    for column in categorical_columns:
        df = alert_label(df, column, indices=outlier_indices)


def alert_label(data: DataFrame, label_name: str, percentage=5, indices=None):
    new_data = data.copy()
    values = data[label_name].unique()
    if indices is None:
        total_number = int(len(data) * percentage / 100)
        indices = np.random.choice(data.index, total_number, replace=False)
    new_data.loc[indices, label_name] = np.random.choice(values)
    return new_data


def generate_point_outliers(df: DataFrame, numerical_columns, categorical_columns, percentage=5, factor=0.5):
    num_outliers = int(len(df) * percentage / 100)
    outlier_indices = np.random.choice(df.index, num_outliers, replace=False)
    outliers = df.loc[outlier_indices].copy()
    outliers[numerical_columns] *= factor
    for column in categorical_columns:
        df = alert_label(df, column, indices=outlier_indices)
    outliers['text'] = 'text'
    return outliers
