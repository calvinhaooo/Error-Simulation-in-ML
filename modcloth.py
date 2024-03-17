from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import *
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

from data_cleaner import textprocess, remove_outliers_iqr
from data_preprocessor import *
from error_generator import *


def define_training_pipeline(numerical_columns, categorical_columns) -> Pipeline:
    print("Setting up training pipeline")
    feature_transformation = ColumnTransformer(transformers=[
        ('numerical_features', StandardScaler(), numerical_columns),
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('textual_features', TfidfVectorizer(), 'text'),
        # ('other_features', 'passthrough', other_columns)
    ], remainder="drop")

    pipeline = Pipeline([
        ('features', feature_transformation),
        # ('classifier', SGDClassifier(loss='log_loss', penalty='l2', max_iter=500))
        ('classifier', SGDClassifier(loss='log_loss'))
        # ('classifier', LogisticRegression(multi_class='multinomial', max_iter=500))
        # ('classifier', KNeighborsClassifier())  # slow
        # ('classifier', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)) # slow
        # ('classifier', BernoulliNB())
        # ('classifier', DecisionTreeClassifier()) # slow
        # ('classifier', SVC())  # slow
        # ('classifier', AdaBoostClassifier()) # slow
    ])

    return pipeline


def run_pipeline(train_set, test_set, numerical_columns, categorical_columns):
    train_x, train_y = train_set
    test_x, test_y = test_set
    print("---------------------------------------")
    sklearn_pipeline = define_training_pipeline(numerical_columns, categorical_columns)
    # train the model
    print("Start training pipeline")
    model = sklearn_pipeline.fit(train_x, train_y)
    # evaluate the model on the test set
    pred = model.predict(test_x)
    prob = model.predict_proba(test_x)
    evaluate(y_scores=prob, y_pred=pred, y_true=test_y)


def evaluate(y_scores, y_pred, y_true):
    auc = roc_auc_score(y_true, y_scores, average='macro', multi_class='ovo')  # or average='micro'/'weighted'
    print("AUC Score:", auc)
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
    return accuracy


if __name__ == '__main__':
    # dataframe = read_data(file_name='modcloth_final_data.json')
    dataframe = read_data(file_name='renttherunway_final_data.json')

    # mc_data = read_data(file_name='dc-crimes-search-results.csv')
    # mc_data = mc_data.drop(columns=['offensekey', 'offense-text', 'offensekey', 'OFFENSE', 'ucr-rank'])
    # label_name = 'NEIGHBORHOOD_CLUSTER'
    # label_name = 'offensegroup'
    label_name = 'fit'

    categories, numerics, texts = select_features(dataframe, label_name, alpha=0.2, max_categories=20)
    print("Categorical columns:", categories)
    print("Numerical columns:", numerics)
    print("Text columns:", texts)

    text_column = 'text'
    merge_text(dataframe, texts, text_column)

    final_columns = categories + numerics + [text_column, label_name]
    print("Final columns:", final_columns)

    # simple clean
    precessed_data = dataframe[final_columns]
    print(len(precessed_data))
    clean_data = precessed_data.dropna()
    clean_data = clean_data.drop_duplicates()
    print(len(clean_data))

    seed = 1234
    test_size = 0.2
    np.random.seed(seed)
    labels = clean_data.pop(label_name)

    train_data, test_data, train_labels, test_labels = train_test_split(
        clean_data, labels, test_size=test_size, random_state=seed)

    dirty_data = train_data.copy()

    add_outliers(dirty_data, numerical_columns=numerics, categorical_columns=categories, outlier_percentage=25,
                 factor=10)
    # add_noise_to_text(dirty_data, percentage=90, noise_percentage=40)
    cleaned_data = dirty_data.copy()
    # textprocess(cleaned_data, 'text')
    for column in numerics:
        remove_outliers_iqr(cleaned_data, column)
    print(cleaned_data)
    # run_pipeline((train_data, train_labels), (test_data, test_labels), numerics, categories)
    # run_pipeline((dirty_data, train_labels), (test_data, test_labels), numerics, categories)

    run_pipeline((cleaned_data, train_labels), (test_data, test_labels), numerics, categories)
