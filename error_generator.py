import random

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


def add_univariate_outliers(df, numerical_columns, categorical_columns, outlier_percentage=5, factor=3):
    num_outliers = int(len(df) * outlier_percentage / 100)
    print(num_outliers)
    outlier_indices = np.random.choice(df.index, num_outliers, replace=False)
    # outliers = np.random.normal(np.mean(df[column]) * factor, np.std(df[column]) * factor, num_outliers)
    # outliers = df.loc[outlier_indices, column]
    df.loc[outlier_indices, numerical_columns] *= factor
    for column in categorical_columns:
        df = alert_label(df, column, indices=outlier_indices)


def add_outliers_random_rows(df, num_rows, numeric_columns, factor):
    rows_indices = np.random.choice(df.index, num_rows, replace=False)
    for row_index in rows_indices:
        for column in numeric_columns:
            outlier = np.random.normal(np.mean(df[column]) * factor, np.std(df[column]) * factor)
            df.loc[row_index, column] += outlier


def alert_label(data: DataFrame, label_name: str, percentage=5, indices=None):
    new_data = data.copy()
    values = data[label_name].unique()
    if indices is None:
        total_number = int(len(data) * percentage / 100)
        indices = np.random.choice(data.index, total_number, replace=False)
    new_data.loc[indices, label_name] = np.random.choice(values)
    return new_data


def generate_multivariate_outliers(df: DataFrame, numerical_columns, categorical_columns, percentage=5, factors=None):
    """
    This function is to generate multivariate outliers in a DataFrame.
    :param df: This is the DataFrame in which the outliers will be generated.
    :param numerical_columns: List of numerical columns in the DataFrame.
    :param categorical_columns: List of categorical columns in the DataFrame.
    :param percentage: The percentage of outliers to generate. Default is 5%.
    :param factors: A list of factors to scale the outliers. Default is [0.5].
    :return: A DataFrame of multivariate outliers.
    """
    # If factors is not provided, set it to default value [0.5]
    if factors is None:
        factors = [0.5]
    # Randomly select indices from the DataFrame according to the percentage
    num_outliers = int(len(df) * percentage / 100)
    outlier_indices = np.random.choice(df.index, num_outliers, replace=False)
    outliers = df.loc[outlier_indices].copy()
    # Scale the numerical columns of the outliers using the random scaling matrix
    random_matrix = np.random.choice(factors, size=outliers[numerical_columns].shape)
    outliers[numerical_columns] *= random_matrix
    # For each categorical column, alert the value randomly
    for column in categorical_columns:
        df = alert_label(df, column, indices=outlier_indices)
    # Set a dummy 'text' column to the outliers DataFrame
    outliers['text'] = 'text'
    return outliers


def modify_labels_to_negative(labels, percentage=25):
    """
    Modify a percentage of labels to negative.

    Args:
    labels (pandas.Series): The series containing labels.
    percentage (float, optional): Percentage of labels to modify. Defaults to 20.

    Returns:
    pandas.Series: Series with modified labels.
    """
    positive_indices = labels[labels == 'positive'].index
    num_to_change = int(len(positive_indices) * percentage / 100)
    indices_to_change = np.random.choice(positive_indices, num_to_change, replace=False)
    labels_modified = labels.copy()
    labels_modified.loc[indices_to_change] = 'negative'
    return labels_modified
