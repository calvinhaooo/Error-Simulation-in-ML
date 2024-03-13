from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype


def select_features(label, data, alpha=0.1, beta=1e-3):
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
