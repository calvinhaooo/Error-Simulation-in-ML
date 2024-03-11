import numpy as np
import random
import string
import pandas as pd
def add_noise(df, column, mean=0):
    mc_data = df.copy(deep=True)
    stddev = np.std(mc_data[column])
    sampled_rows = mc_data.sample(frac=0.3)
    noise = np.random.normal(mean, stddev, size=len(sampled_rows))
    df.loc[sampled_rows.index, column] += noise

def add_outliers(df, column, outlier_percentage=5, factor=3):
    num_outliers = int(len(df) * outlier_percentage / 100)
    outlier_indices = np.random.choice(df.index, num_outliers, replace=False)
    outliers = np.random.normal(np.mean(df[column]) * factor, np.std(df[column]) * factor, num_outliers)
    df.loc[outlier_indices, column] += outliers

def add_duplicates(df, duplicate_percentage=5):
    """
    Add duplicate rows to a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame.
        duplicate_percentage (float, optional): Percentage of duplicate rows to add. Defaults to 5.

    Returns:
        None
    """
    num_duplicates = int(len(df) * duplicate_percentage / 100)
    duplicate_rows = df.sample(n=num_duplicates, replace=True)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

# def add_noise_to_text(text, noise_percentage=10):
#     words = text.split()
#     num_noise_words = int(len(words) * noise_percentage / 100)
#     noise_words = ['@#$%&'] * num_noise_words
#     noisy_text = ' '.join(words[:num_noise_words] + noise_words + words[num_noise_words:])
#     return noisy_text

def add_noise_to_text(text, noise_percentage=10):
    words = text.split()
    num_noise_chars = int(len(text) * noise_percentage / 100)

    # characters = ['~!@#$%^&*()'] + string.punctuation
    characters = string.punctuation
    for _ in range(num_noise_chars):
        index = random.randint(0, len(text) - 1)
        noise_char = random.choice(characters)
        text = text[:index] + noise_char + text[index:]
    return text

def add_null_noise(df, label_column, null_percentage=5):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_modify = [col for col in numeric_cols if col != label_column]
    num_nulls = int(len(df) * null_percentage / 100)
    # 随机选择将要设置为空值的索引
    null_indices = np.random.choice(df.index, num_nulls, replace=False)
    # 对于每一个选中的索引，随机选择一个数值列名并设置该位置的值为NaN
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

