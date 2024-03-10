from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype


def select_features(label, data, alpha=0.1):
    categorical_columns = []
    numerical_columns = []
    text_columns = []
    for col in data.columns:
        data_type = data[col].dtype
        null_num = data[col].isnull().sum()
        feature_num = data[col].nunique()
        if null_num / len(data) > alpha or col == label:
            print(col, null_num / len(data))
            continue

        if is_numeric_dtype(data_type):
            numerical_columns.append(col)
        elif feature_num <= len(data) * 0.001:
            categorical_columns.append(col)
        elif is_string_dtype(data_type):
            text_columns.append(col)

    return categorical_columns, numerical_columns, text_columns
