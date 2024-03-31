from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data_preprocessor import *
from error_generator import *
from data_cleaner import *

def define_training_pipeline(numerical_columns, categorical_columns) -> Pipeline:
    print("Setting up training pipeline")
    feature_transformation = ColumnTransformer(transformers=[
        ('numerical_features', StandardScaler(), numerical_columns),
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('textual_features', CountVectorizer(), 'text'),
    ], remainder="drop")

    pipeline = Pipeline([
        ('features', feature_transformation),
        ('classifier', BernoulliNB())
    ])

    return pipeline


def run_pipeline(train_set, test_set, numerical_columns, categorical_columns, task_type):
    train_x, train_y = train_set
    test_x, test_y = test_set
    print("---------------------------------------")
    sklearn_pipeline = define_training_pipeline(numerical_columns, categorical_columns)
    # train the model
    print("Start training pipeline")
    model = sklearn_pipeline.fit(train_x, train_y)
    # evaluate the model on the test set
    if task_type == 'classification':
        pred = model.predict(test_x)
        prob = model.predict_proba(test_x)
        evaluate(y_scores=prob, y_pred=pred, y_true=test_y)
    elif task_type == 'regression':
        pred = model.predict(test_x)
        evaluatePredict(y_pred=pred, y_true=test_y)
    else:
        raise ValueError("Unknown task type: {}".format(task_type))


def evaluate(y_scores, y_pred, y_true):
    # auc = roc_auc_score(y_true, y_scores, average='macro', multi_class='ovo')  # or average='micro'/'weighted'
    # print("AUC Score:", auc)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    precision = precision_score(y_true, y_pred, average='macro')  # or average='micro'/'weighted'
    print("Precision:", precision)
    recall = recall_score(y_true, y_pred, average='macro')  # or average='micro'/'weighted'
    print("Recall:", recall)
    f1 = f1_score(y_true, y_pred, average='macro')  # or average='micro'/'weighted'
    print("F1 Score:", f1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    if len(y_scores.shape) == 2:  # 如果y_scores是二维的，假定第二列是正类的概率
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1], pos_label='positive')
    else:
        fpr, tpr, _ = roc_curve(y_true, y_scores)

    return accuracy

def evaluatePredict(y_pred, y_true):
    mse = mean_squared_error(y_true, y_pred)
    print("MSE:", mse)
    rmse = mse ** 0.5
    print("RMSE:", rmse)
    r2 = r2_score(y_true, y_pred)
    print("R² Score:", r2)

def plot_label_distribution(labels, title):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    labels.value_counts().plot(kind='bar')
    plt.show()

if __name__ == '__main__':
    mc_data = read_data(file_name='IMDB Dataset.csv')
    label_name = 'sentiment'
    # task is divided into regression or classification
    task_type = 'classification'

    categories, numerics, texts = select_features(mc_data, label_name, alpha=0.2)
    print("Categorical columns:", categories)
    print("Numerical columns:", numerics)
    print("Text columns:", texts)

    text_column = 'text'
    merge_text(mc_data, texts, text_column)

    final_columns = categories + numerics + [text_column, label_name]
    print("Final columns:", final_columns)

    # simple clean
    precessed_data = mc_data[final_columns]
    # print(len(precessed_data))
    clean_data = precessed_data.dropna()
    clean_data = clean_data.drop_duplicates()
    # print(len(clean_data))

    seed = 1234
    test_size = 0.2
    print(clean_data.columns)
    labels = clean_data.pop(label_name)

    train_data, test_data, train_labels, test_labels = train_test_split(
        clean_data, labels, test_size=test_size, random_state=seed)

    positive_before = (train_labels == 'positive').sum()
    negative_before = (train_labels == 'negative').sum()
    print("train_label length", len(train_labels))
    train_labels_modified = modify_labels_to_negative(train_labels)
    positive_after = (train_labels_modified == 'positive').sum()
    negative_after = (train_labels_modified == 'negative').sum()
    # label imbalance --> 40% pos to neg
    print(f"\nBefore modification: Positive: {positive_before}, Negative: {negative_before}")
    print(f"\nafter modification: Positive: {positive_after}, Negative: {negative_after}")

    plot_label_distribution(train_labels, "Before Modification")
    plot_label_distribution(train_labels_modified, "After Modification")

    train_labels_clean_str = clean_bc_label_error(train_data, train_labels)
    positive_clean_before = (train_labels_clean_str == 'positive').sum()
    negative_clean_before = (train_labels_clean_str == 'negative').sum()
    print(f"\nafter cleaned: Positive: {positive_clean_before}, Negative: {negative_clean_before}")

    train_labels_clean_str = pd.Series(train_labels_clean_str, index=train_labels_modified.index)
    print(train_labels_clean_str)
    plot_label_distribution(train_labels_clean_str, "after cleaned")


    run_pipeline((train_data, train_labels), (test_data, test_labels), numerics, categories, task_type)
    run_pipeline((train_data, train_labels_modified), (test_data, test_labels), numerics, categories, task_type)
    run_pipeline((train_data, train_labels_clean_str), (test_data, test_labels), numerics, categories, task_type)
