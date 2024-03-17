import random
import string

import numpy as np
import pandas as pd
from pandas import DataFrame


def add_noise(df, column, mean=0):
    if df[column].dtypes == 'float64':
        mc_data = df.copy(deep=True)
        stddev = np.std(mc_data[column])
        sampled_rows = mc_data.sample(frac=0.3)
        noise = np.random.normal(mean, stddev, size=len(sampled_rows))
        df.loc[sampled_rows.index, column] += noise


def add_outliers(df, numerical_columns, categorical_columns, outlier_percentage=5, factor=3):
    num_outliers = int(len(df) * outlier_percentage / 100)
    outlier_indices = np.random.choice(df.index, num_outliers, replace=False)
    # outliers = np.random.normal(np.mean(df[column]) * factor, np.std(df[column]) * factor, num_outliers)
    # outliers = df.loc[outlier_indices, column]
    df.loc[outlier_indices, numerical_columns] *= factor
    for column in categorical_columns:
        df = alert_label(df, column, indices=outlier_indices)


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
    # return tex


def alert_label(data: DataFrame, label_name: str, percentage=5, indices=None):
    new_data = data.copy()
    values = data[label_name].unique()
    if indices is None:
        total_number = int(len(data) * percentage / 100)
        indices = np.random.choice(data.index, total_number, replace=False)
    new_data.loc[indices, label_name] = np.random.choice(values)
    return new_data


def duplicates_data(data: DataFrame, percentage=3):
    total_number = int(len(data) * percentage / 100)
    random_rows = data.sample(n=total_number)
    random_rows = DataFrame(random_rows)
    new_data = pd.concat([data, random_rows])
    return new_data
