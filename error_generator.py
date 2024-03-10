import numpy as np


def add_noise(df, column, mean=0):
    if df[column].dtypes == 'float64':
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
