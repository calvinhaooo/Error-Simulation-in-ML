from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype


def select_features(label, data):
    categorical_columns = []
    numerical_columns = []
    text_columns = []
    for col in data.columns:
        data_type = data[col].dtype
        null_num = data[col].isnull().sum()
        feature_num = data[col].nunique()
        if null_num / len(data) > 0.1 or col == label or feature_num >= 0.9 * len(data):
            print(col, null_num / len(data))
            continue

        if is_numeric_dtype(data_type):
            numerical_columns.append(col)
        elif feature_num <= len(data) * 0.001:
            categorical_columns.append(col)
        elif is_string_dtype(data_type):
            text_columns.append(col)

    print(categorical_columns)
    print(numerical_columns)
    print(text_columns)
    return categorical_columns, numerical_columns, text_columns
