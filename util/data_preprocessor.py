import pandas as pd
from pandas import DataFrame, isnull
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype


def read_data(file_name):
    path = "./data/" + file_name
    if file_name.endswith('.csv'):
        data = pd.read_csv(path)
    elif file_name.endswith('.json'):
        data = pd.read_json(path, lines=True)
    else:
        raise ValueError("Unsupported file")
    return data


def select_features(data, label, alpha=0.1, beta=1e-3, max_categories=10):
    categorical_columns, numerical_columns, text_columns = [], [], []
    for column in data.columns:
        # skip label column
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
        # if is_numeric_dtype(data_type) and feature_num > max_categories:
        if is_numeric_dtype(data_type):
            numerical_columns.append(column)
        elif feature_num <= len(data) * beta:
            categorical_columns.append(column)
        elif is_string_dtype(data_type):
            text_columns.append(column)

    return categorical_columns, numerical_columns, text_columns


def merge_text(data: DataFrame, text_columns, column_name):
    data[column_name] = ''
    for text_column in text_columns:
        data[column_name] += ' ' + data[text_column]


def convert_cup_size_to_cms(cup_size_code):
    if cup_size_code == 'aa':
        return 10.5
    if cup_size_code == 'a':
        return 12.5
    if cup_size_code == 'b':
        return 14.5
    if cup_size_code == 'c':
        return 16.5
    if cup_size_code == 'd':
        return 18.5
    if cup_size_code == 'dd/e':
        return 20.5
    if cup_size_code == 'ddd/f':
        return 22.5
    if cup_size_code == 'dddd/g':
        return 24.5
    if cup_size_code == 'h':
        return 26.5
    if cup_size_code == 'i':
        return 28.5
    if cup_size_code == 'j':
        return 30.5
    if cup_size_code == 'k':
        return 32.5
    else:
        return None


def convert_height_to_number(height_in_inch):
    if isnull(height_in_inch):
        return None
    words = height_in_inch.split()
    ft = int(words[0][0])
    if len(words) == 2:
        inch = int(words[1][0])
    else:
        inch = 0
    return ft * 12 + inch


def parse_date(df, column_name, date=True, time=False, weekday=False):
    df[column_name] = pd.to_datetime(df[column_name])
    if date:
        df['year'] = df[column_name].dt.year
        df['month'] = df[column_name].dt.month
        df['day'] = df[column_name].dt.day
    if time:
        df['hour'] = df[column_name].dt.hour
        df['minute'] = df[column_name].dt.minute
        df['second'] = df[column_name].dt
    if weekday:
        df['weekday'] = df[column_name].dt.weekday
