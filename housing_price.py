from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import *
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from data_preprocessor import *
from error_generator import *
from data_cleaner import *

def define_training_pipeline(numerical_columns, categorical_columns) -> Pipeline:
    print("Setting up training pipeline")
    feature_transformation = ColumnTransformer(transformers=[
        ('numerical_features', StandardScaler(), numerical_columns),
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        # ('textual_features', CountVectorizer(), 'text'),
        # ('other_features', 'passthrough', other_columns)
    ], remainder="drop")

    pipeline = Pipeline([
        ('features', feature_transformation),
        # ('classifier', SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, random_state=seed))
        # ('classifier', LogisticRegression(multi_class='multinomial', max_iter=100))
        # ('classifier', KNeighborsClassifier())
        # ('classifier', KNeighborsClassifier())  # slow
        # ('classifier', DecisionTreeClassifier(max_depth=50, random_state=40))
        ('classifier', LinearRegression())
        # ('classifier', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)) # slow
        # ('classifier', BernoulliNB())
        # ('classifier', DecisionTreeClassifier()) # slow
        # ('classifier', SVC())  # slow
        # ('classifier', AdaBoostClassifier()) # slow
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
        # prob = model.predict_proba(test_x)[:, 1]
        prob = model.predict_proba(test_x)
        evaluate(y_scores=prob, y_pred=pred, y_true=test_y)
    elif task_type == 'regression':
        pred = model.predict(test_x)
        evaluatePredict(y_pred=pred, y_true=test_y)
    else:
        raise ValueError("Unknown task type: {}".format(task_type))


def evaluate(y_scores, y_pred, y_true):
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

def evaluatePredict(y_pred, y_true):
    mse = mean_squared_error(y_true, y_pred)
    print("MSE:", mse)
    rmse = mse ** 0.5
    print("RMSE:", rmse)
    r2 = r2_score(y_true, y_pred)
    print("R² Score:", r2)

if __name__ == '__main__':
    mc_data = read_data(file_name='housing_price_dataset.csv')
    label_name = 'Price'
    task_type = 'regression'

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
    np.random.seed(seed)

    labels = clean_data.pop(label_name)

    train_data, test_data, train_labels, test_labels = train_test_split(
        clean_data, labels, test_size=test_size, random_state=seed)

    dirty_data = train_data.copy()
    # add_outliers(dirty_data, column='SquareFeet', outlier_percentage=20, factor= 5)
    add_outliers_random_rows(dirty_data, num_rows= 4000, numeric_columns=numerics, factor=10)
    # add_noise_to_text(dirty_data, percentage= 50, noise_percentage= 30)
    # print(dirty_data['text'])
    clean_data = dirty_data.copy()
    clean_data = remove_outliers_iqr(clean_data, numerics)
    detectN_impute_KNN(clean_data)
    # clean_data_2 = detectN_impute_KNN(clean_data, numerics)
    # A 变成拉丁
    # dirty_data = add_broken_characters(dirty_data, column='text', fraction=0.5)
    # print(dirty_data.columns)

    # label error
    # dirty_train_label = train_labels.copy()
    # dirty_test_label = test_labels.copy()
    # random_replace_labels(dirty_train_label, dirty_test_label, num_labels= 3)
    # print(dirty_label.unique())

    # 加入噪声影响不大
    # dirty_data = add_gaussian_noise(dirty_data, column='SquareFeet', fraction=0.5)

    run_pipeline((train_data, train_labels), (test_data, test_labels), numerics, categories, task_type)
    run_pipeline((dirty_data, train_labels), (test_data, test_labels), numerics, categories, task_type)
    run_pipeline((clean_data, train_labels), (test_data, test_labels), numerics, categories, task_type)
