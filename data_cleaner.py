import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def textprocess(df, column):
    pattern = r'[^a-zA-Z0-9\s]+'
    df[column] = df[column].apply(lambda x: re.sub(pattern, '', x).strip())


# ??
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print("Identified outliers count: ", len(outliers))
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), np.nan, df[column])
    return df
    # df.drop(df[(df[column] < lower_bound) | (df[column] > upper_bound)].index, inplace=True)


def detect_impute_knn(df):
    # detect which colum has Null
    missing_values_count = df.isnull().sum()
    print("The number of Null for each column：")
    print(missing_values_count)
    print("\nThe number of all null:", missing_values_count.sum())
    imputer = KNNImputer(n_neighbors=50)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        imputed_data = imputer.fit_transform(df[numeric_cols])
        df[numeric_cols] = imputed_data
    else:
        print("There is no numeric columns we can fill in")


def remove_noise(df, column, kernel_size=3):
    df[column] = signal.medfilt(df[column].values, kernel_size=kernel_size)


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


def remove_duplication(df, method='simple'):
    new_df = df.copy()
    if method == 'simple':
        new_df = new_df.drop_duplicates()
    elif method == 'similarity':
        pass
        # todo
    return new_df


def cluster_labels(df, label):
    features = df['text']
    labels = df[label]

    # Encoding labels if they are categorical
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    text_vectorizer = TfidfVectorizer()
    encoded_text = text_vectorizer.fit_transform(features)

    # use TruncatedSVD to extract main features
    n_components = 50
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_text = svd.fit_transform(encoded_text)

    # Creating k-NN classifier without specifying the number of neighbors
    knn = KNeighborsClassifier()
    knn.fit(reduced_text, encoded_labels)
    new_labels = knn.predict(reduced_text)
    print(f"Number of label types after clustering is {len(set(new_labels))}.")

    new_df = df.copy()
    new_df[label] = new_labels
    return new_df


def merge_outliers(original_data, outlier_data, feature_transformer, visualization=False):
    # print(outlier_data)
    outliers_num = len(outlier_data)
    merged_data = pd.concat([outlier_data, original_data])
    transformed_data = feature_transformer.fit_transform(merged_data)

    if visualization:
        # Dimensionality Reduction
        svd = TruncatedSVD(n_components=2)
        reduced_data = svd.fit_transform(transformed_data)
        # visualization
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[outliers_num:, 0], reduced_data[outliers_num:, 1], s=1e3 / outliers_num, c='blue',
                    alpha=0.02)
        plt.scatter(reduced_data[:outliers_num, 0], reduced_data[:outliers_num, 1], s=1e4 / outliers_num, c='red',
                    alpha=0.3)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('SVD Reduced Data')
        plt.grid(True)
        plt.show()

    return merged_data, transformed_data


def detect_point_outliers(transformed_data, n=6, percentile=1, visualization=False):
    svd = TruncatedSVD(n_components=n)
    reduced_data = svd.fit_transform(transformed_data)

    isolation_forest = IsolationForest()
    isolation_forest.fit(reduced_data)
    scores = isolation_forest.decision_function(reduced_data)
    if visualization:
        # Draw Score Distribution
        plt.hist(scores, bins=int(len(scores) / 600), color='blue', alpha=0.7)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution')
        plt.show()

    threshold = np.percentile(scores, percentile)
    print('threshold score:', threshold)
    outlier_pos = np.where(scores < threshold)

    return outlier_pos[0]
