import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import BaggingClassifier
from error_solution import *
from data_preprocessor import select_features
from error_generator import *


def define_training_pipeline(numerical_columns, categorical_columns):
    print("Setting up training pipeline")
    feature_transformation = ColumnTransformer(transformers=[
        ('numerical_features', StandardScaler(), numerical_columns),
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('textual_features', CountVectorizer(), 'text'),
    ], remainder="drop")

    pipeline = Pipeline([
        ('features', feature_transformation),
        # ('learner', SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000))
        ('learner', LogisticRegression(multi_class='multinomial', max_iter=100))
        # ('classifier', KNeighborsClassifier())  # 又慢又不准
        # ('classifier', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)) # laji
        # ('learner', BaggingClassifier())
    ])

    return pipeline


def run_pipeline(data, label, seed, test_size):
    np.random.seed(seed)
    # data = mc_data[final_columns].dropna()
    print(len(data))
    labels = data.pop(label)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=seed)
    sklearn_pipeline = define_training_pipeline(numerical_columns, categorical_columns)
    model_with_transformations = sklearn_pipeline.fit(train_data, train_labels)
    pred = model_with_transformations.predict(test_data)
    prob = model_with_transformations.predict_proba(test_data)

    # y_true = test_labels
    evaluate(y_scores=prob, y_pred=pred, y_true=test_labels)
    # accuracy = model_with_transformations.score(test_data, test_labels)
    # print(f"Accuracy on test data is {accuracy}")

    # return accuracy


def evaluate(y_scores, y_pred, y_true):
    auc = roc_auc_score(y_true, y_scores, average='macro', multi_class='ovo')  # or average='micro'/'weighted'
    print("AUC Score:", auc)
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    # 计算精确度
    precision = precision_score(y_true, y_pred, average='macro')  # or average='micro'/'weighted'
    print("Precision:", precision)
    # 计算召回率
    recall = recall_score(y_true, y_pred, average='macro')  # or average='micro'/'weighted'
    print("Recall:", recall)
    # 计算F1分数
    f1 = f1_score(y_true, y_pred, average='macro')  # or average='micro'/'weighted'
    print("F1 Score:", f1)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    return accuracy


if __name__ == '__main__':
    # ./data/modcloth_final_data.json
    # ./data/renttherunway_final_data.json
    # mc_data = pd.read_json('./data/Magazine_Subscriptions.json', lines=True)
    mc_data = pd.read_json('./data/modcloth_final_data.json', lines=True)

    # label_name = 'overall'
    label_name = 'fit'
    # mc_data[label_name] = OneHotEncoder().fit_transform(mc_data[label_name])
    # mc_data = pd.get_dummies(mc_data, columns=[label_name])
    # print(mc_data)

    # for amazon review
    # mc_data.drop(columns=['vote','style','image'], inplace= True)
    print(mc_data.info())
    # mc_data.drop(columns=['vote', 'image', 'style'], inplace=True)
    #
    # print(mc_data.info())
    # print(mc_data.corr())

    categorical_columns, numerical_columns, text_columns = select_features(label_name, mc_data, alpha=0.2)
    print(categorical_columns)
    print(numerical_columns)
    print(text_columns)
    # numerical_columns.remove('bra size')
    # numerical_columns = []
    # categorical_columns = []

    mc_data['text'] = ''
    for text_column in text_columns:
        mc_data['text'] += ' ' + mc_data[text_column]

    final_columns = categorical_columns + numerical_columns + ['text', label_name]
    # final_columns.remove('user_id')
    print(final_columns)
    precessed_data = mc_data[final_columns]
    cleaned_data = precessed_data.dropna()
    print(cleaned_data.info())
    dirty_data = cleaned_data.copy()
    run_pipeline(cleaned_data, label_name, 1234, 0.2)
    # add_noise(dirty_data, 'verified', mean= 1)
    # 添加异常值
    add_outliers(dirty_data, 'bra size', outlier_percentage=20, factor=3)
    # 添加文本错误
    # dirty_data['text'] = dirty_data['text'].apply(add_noise_to_text, noise_percentage=20)
    # 添加空值
    # add_null_noise(dirty_data, label_name, null_percentage= 10)
    # print(dirty_data.isnull().sum())

    # print(dirty_data['text'])
    clean_data = dirty_data.copy()
    run_pipeline(dirty_data, label_name, 1234, 0.2)

    #clean
    # 修正文本
    # textprocess(clean_data, 'text')
    # print(clean_data.text.sample(10))
    # print(cleaned_data['text'] == clean_data['text'])
    # 这是修正缺失值
    # detectN_impute_KNN(dirty_data, numerical_columns)
    # print("after clean:", dirty_data.isnull().sum())
    # 修正outlier
    # 会把所有的都当作离群点
    remove_outliers_iqr(clean_data, 'bra size')
    print(clean_data.head())
    run_pipeline(clean_data, label_name, 1234, 0.2)

# baggingclassifier
# clean
# AUC Score: 0.8048359379814853
# Accuracy: 0.7407965194109772
# Precision: 0.6263851939688843
# Recall: 0.5311871270904984
# F1 Score: 0.5594924789217963
# dirty add noise
# AUC Score: 0.804939665684558
# Accuracy: 0.7406291834002677
# Precision: 0.6215810053443296
# Recall: 0.5280722005444906
# F1 Score: 0.5567672521119243


# LR
# AUC Score: 0.8607143238972818
# Accuracy: 0.7658969210174029
# Precision: 0.6233366901614423
# Recall: 0.574536575417123
# F1 Score: 0.5925804046136496
# add noise
# AUC Score: 0.8572953324328573
# Accuracy: 0.7617692994199018
# Precision: 0.6149274886331152
# Recall: 0.56715412891674
# F1 Score: 0.5855643027718003

# add outlier
# AUC Score: 0.8474261255810827 -2.7
# Accuracy: 0.7467648371262829 -1.9
# Precision: 0.6035272651773218
# Recall: 0.5370934364570092
# F1 Score: 0.5613565174394379

# add special charater
# AUC Score: 0.8001508916188174
# Accuracy: 0.714859437751004
# Precision: 0.5420861087601131
# Recall: 0.5062115288264054
# F1 Score: 0.5203277656546945

# add outlier to unixReviewTime
# AUC Score: 0.8598201941193266
# Accuracy: 0.7652833556448014
# Precision: 0.6238998373795301
# Recall: 0.5753129446218851
# F1 Score: 0.5937036153666082

# clean
# AUC Score: 0.861704960394787
# Accuracy: 0.7709341545681185
# Precision: 0.6291212030374973
# Recall: 0.5821719660223292
# F1 Score: 0.5990906639006213