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

from data_preprocessor import select_features
from error_generator import add_outliers


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
    mc_data = pd.read_json('./data/modcloth_final_data.json', lines=True)
    label_name = 'fit'
    # mc_data[label_name] = OneHotEncoder().fit_transform(mc_data[label_name])
    # mc_data = pd.get_dummies(mc_data, columns=[label_name])
    # print(mc_data)

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

    dirty_data = cleaned_data.copy()
    run_pipeline(cleaned_data, label_name, 1234, 0.2)

    add_outliers(dirty_data, 'size', outlier_percentage=10, factor=3)
    run_pipeline(dirty_data, label_name, 1234, 0.2)

# without bra size
# AUC Score: 0.8514183711474294
# Accuracy: 0.7881227981882235
# Precision: 0.7380110362275406
# Recall: 0.6226882933203429
# F1 Score: 0.6628743242436715
# Confusion Matrix:
# [[8924  321  288]
#  [1105  968  135]
#  [ 977  121 1070]]

# cleaned data
# AUC Score: 0.8518708069803583
# Accuracy: 0.7859302995391705
# Precision: 0.7237415956573389
# Recall: 0.6371173298094482
# F1 Score: 0.6692058156810673
# Confusion Matrix:
# [[8731  350  392]
#  [1044  990  181]
#  [ 882  124 1194]]

# add outlier bra size
# AUC Score: 0.8511242361643943
# Accuracy: 0.7862183179723502
# Precision: 0.7265108824576934
# Recall: 0.6347286331629108
# F1 Score: 0.6677928386486712
# Confusion Matrix:
# [[8757  323  393]
#  [1074  961  180]
#  [ 880  119 1201]]

# log, only text
# AUC Score: 0.8298759894064216
# Accuracy: 0.7758496023138105
# Precision: 0.6936738196258047
# Recall: 0.6172401446556065
# F1 Score: 0.6465030296115503
# Confusion Matrix:
# [[9608  490  431]
#  [1164 1052  195]
#  [ 965  165 1143]]

# AUC Score: 0.8460666033583649
# Accuracy: 0.7785858294930875
# Precision: 0.7076563476990462
# Recall: 0.6382360424106194
# F1 Score: 0.6658485754518694
# Confusion Matrix:
# [[8588  473  412]
#  [ 998 1050  167]
#  [ 890  135 1175]]

