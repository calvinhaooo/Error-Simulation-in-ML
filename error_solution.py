import re
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


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
        print("DataFrame中没有数值型列可以填补。")

# textprocess(mc_data,'review_summary')
# mc_data.review_summary.sample(10)