import numpy as np
import random
import string

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


def insert_special_word(text, insert_percentage=5):
    special_word = '@#$%&'
    words = text.split()
    num_to_insert = int(len(words) * insert_percentage / 100)
    positions_to_insert = np.random.choice(range(len(words)), num_to_insert, replace=False)
    for pos in positions_to_insert:
        words.insert(pos, special_word)
    modified_text = ' '.join(words)
    return modified_text

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
    # num_nulls = int(len(df) * null_percentage / 100)
    # null_indices = np.random.choice(df.index, num_nulls, replace=False)
    # for index in null_indices:
    #     col = np.random.choice(df.columns)
    #     df.at[index, col] = np.nan