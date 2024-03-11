import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def textprocess(df,column):
    pattern=r'[^a-zA-Z0-9\s]+'
    df[column]=df[column].apply(lambda x: re.sub(pattern, '', x).strip())

# ??
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print("Identified outliers count: ", len(outliers))
    df.drop(df[(df[column] < lower_bound) | (df[column] > upper_bound)].index, inplace=True)


def detectN_impute_KNN(df,numerical_columns):
    # detect which colum has Null
    missing_values_count = df.isnull().sum()
    print("The number of Null for each column：")
    print(missing_values_count)
    print("\nThe number of all null:", missing_values_count.sum())
    imputer = KNNImputer(n_neighbors=5)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        # 对数值型数据应用KNN填补
        imputed_data = imputer.fit_transform(df[numeric_cols])
        # 更新原始DataFrame的数值型列
        df[numeric_cols] = imputed_data
    else:
        print("There is no numeric columns we can fill in")

def detect_duplicates(df):
    """
    Detect duplicate rows in a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to check for duplicates.

    Returns:
        pandas.DataFrame: DataFrame containing duplicate rows.
    """
    duplicate_rows = df[df.duplicated()]
    return duplicate_rows

def correct_label_errors(data, label_name, features_columns):
    logistic_regression = LogisticRegression()
    knn_classifier = KNeighborsClassifier()
    random_forest = RandomForestClassifier()

    # 获取特征和标签
    features = data[features_columns]
    encoded_labels = data[label_name]

    # 使用LabelEncoder将标签转换为整数类型
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(encoded_labels)
    # 创建 VotingClassifier 实例
    voting_classifier = VotingClassifier(estimators=[
        ('logistic_regression', logistic_regression),
        ('knn', knn_classifier),
        ('random_forest', random_forest)
    ], voting='hard')  # 使用硬投票进行决策

    # 使用 VotingClassifier 进行训练
    voting_classifier.fit(features, encoded_labels)

    # 预测结果
    predicted_labels = voting_classifier.predict(features)

    # 将预测结果添加回原始数据
    data[label_name] = predicted_labels

    return data
