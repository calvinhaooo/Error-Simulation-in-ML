
import numpy as np
import pandas as pd
from datetime import timedelta

def introduce_missing_values(df, column, missing_percentage=10):
    """
    Randomly introduces missing values into a specified column of the DataFrame.
    """
    num_missing = int(len(df) * missing_percentage / 100)
    missing_indices = np.random.choice(df.index, num_missing, replace=False)
    df.loc[missing_indices, column] = np.nan
    return df

def add_outliers(df, column, outlier_percentage=5, factor=3):
    num_outliers = int(len(df) * outlier_percentage / 100)
    outlier_indices = np.random.choice(df.index, num_outliers, replace=False)
    outliers = np.random.normal(np.mean(df[column]) * factor, np.std(df[column]) * factor, num_outliers)
    df.loc[outlier_indices, column] += outliers

def introduce_duplicates(df, duplicate_percentage=5):
    """
    Introduces duplicate rows into the DataFrame.
    """
    if df.empty:
        return df  # Return the DataFrame unchanged if it's empty
    
    num_duplicates = int(len(df) * duplicate_percentage / 100)
    duplicate_indices = np.random.choice(df.index, num_duplicates, replace=False)
    duplicates = df.loc[duplicate_indices]
    
    # Concatenate the original DataFrame with the duplicates
    df = pd.concat([df, duplicates], ignore_index = True).sample(frac=1).reset_index(drop=True)
    
    return df

def introduce_timestamp_errors(df, timestamp_column = 'timestamp', error_percentage=5, max_day_shift=30):
    """
    Introduces errors in the timestamp data by randomly shifting the timestamps.

    Parameters:
    - df: DataFrame containing the timestamp column.
    - timestamp_column: The name of the timestamp column.
    - error_percentage: Percentage of rows to introduce timestamp errors in.
    - max_day_shift: Maximum number of days to shift the timestamp, both in the past and future.

    Returns:
    - Modified DataFrame with errors introduced in the timestamp column.
    """
    num_errors = int(len(df) * error_percentage / 100)
    error_indices = np.random.choice(df.index, num_errors, replace=False)
    for idx in error_indices:
        days_to_shift = np.random.randint(-max_day_shift, max_day_shift + 1)
        time_delta = timedelta(days=days_to_shift)
        df.at[idx, timestamp_column] += time_delta
    
    return df

def introduce_label_noise(df, label_column, noise_percentage=5):
    """
    Randomly changes the labels in the label_column to simulate label noise.
    """
    unique_labels = df[label_column].unique()
    num_noisy_labels = int(len(df) * noise_percentage / 100)
    noisy_indices = np.random.choice(df.index, num_noisy_labels, replace=False)
    
    for idx in noisy_indices:
        current_label = df.loc[idx, label_column]
        possible_labels = [label for label in unique_labels if label != current_label]
        df.loc[idx, label_column] = np.random.choice(possible_labels)
    return df

def add_random_noise(df, column, noise_level=0.1):
    """
    Adds random noise to a numerical column. The noise level is relative to the standard deviation of the column.
    """
    noise = np.random.normal(0, np.std(df[column]) * noise_level, len(df))
    df[column] += noise
    return df

def corrupt_categorical_data(df, column, corruption_percentage=5):
    unique_values = df[column].dropna().unique()
    if len(unique_values) < 2:
        print("Not enough unique values to corrupt.")
        return df
    
    num_rows = len(df)
    num_corrupt = int(num_rows * corruption_percentage / 100)
    indices_to_corrupt = np.random.choice(df.index, num_corrupt, replace=False)
    
    for idx in indices_to_corrupt:
        current_value = df.at[idx, column]
        possible_values = [val for val in unique_values if val != current_value]
        df.at[idx, column] = np.random.choice(possible_values)
    
    return df

def introduce_missing_values(data, columns, missing_percentage):
    """
    Introduce missing values into specified columns of a dataframe.

    Parameters:
    data (DataFrame): The input dataframe.
    columns (list): List of columns to introduce missing values into.
    missing_percentage (float): The percentage of values to be replaced with NaN.

    Returns:
    DataFrame: The dataframe with missing values introduced.
    """
    corrupted_data = data.copy()
    for column in columns:
        total_values = len(corrupted_data[column])
        missing_count = int(total_values * missing_percentage)
        missing_indices = np.random.choice(total_values, missing_count, replace=False)
        corrupted_data.loc[missing_indices, column] = np.nan
    
    return corrupted_data

def introduce_outliers(data, columns, outlier_percentage, outlier_multiplier=10):
    """
    Introduce outliers into specified columns of a dataframe.

    Parameters:
    data (DataFrame): The input dataframe.
    columns (list): List of columns to introduce outliers into.
    outlier_percentage (float): The percentage of values to be replaced with outliers.
    outlier_multiplier (float): The factor to multiply the standard deviation to determine the outlier's value.

    Returns:
    DataFrame: The dataframe with outliers introduced.
    """
    corrupted_data = data.copy()
    for column in columns:
        # Determine the number of outliers to add
        total_values = len(corrupted_data[column])
        outlier_count = int(total_values * outlier_percentage)

        # Calculate the outlier value
        mean = corrupted_data[column].mean()
        std = corrupted_data[column].std()
        outlier_value = mean + outlier_multiplier * std

        # Randomly select indices to replace with outliers
        outlier_indices = np.random.choice(total_values, outlier_count, replace=False)
        corrupted_data.loc[outlier_indices, column] = outlier_value
    
    return corrupted_data