import time
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_extraction.text import *
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from data_cleaner import merge_outliers, detect_multivariate_outliers
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
        ('reduction', TruncatedSVD(n_components=64)),
        # ('classifier', SGDClassifier(loss='log_loss', penalty='l2', max_iter=500))
        ('classifier', SGDClassifier(loss='log_loss', random_state=42))
        # ('classifier', LogisticRegression(multi_class='multinomial', max_iter=500))
        # ('classifier', KNeighborsClassifier())  # slow
        # ('classifier', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100))  # slow
        # ('classifier', BernoulliNB())
        # ('classifier', DecisionTreeClassifier())  # slow
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
    warnings.filterwarnings("ignore", category=DataConversionWarning)

    # dataframe = read_data(file_name='modcloth_final_data.json')
    dataframe = read_data(file_name='renttherunway_final_data.json')

    label_name = 'fit'

    parse_date(dataframe, 'review_date')
    dataframe['weight'] = pd.to_numeric(dataframe['weight'].str.replace('lbs', '').astype(float))
    dataframe['height'] = dataframe['height'].apply(convert_height_to_number)

    # dataframe['height'] = dataframe['height'].apply(convert_height_to_number)
    # dataframe['cup size'] = dataframe['cup size'].apply(convert_cup_size_to_cms)

    categories, numerics, texts = select_features(dataframe, label_name, alpha=0.2, max_categories=12)
    print("Categorical columns:", categories)
    print("Numerical columns:", numerics)
    print("Text columns:", texts)

    text_column = 'text'
    merge_text(dataframe, texts, text_column)

    final_columns = categories + numerics + [text_column, label_name]
    print("Final columns:", final_columns)

    # simple clean
    precessed_data = dataframe[final_columns]
    print('Original rows:', len(precessed_data))
    clean_data = precessed_data.dropna()
    clean_data = clean_data.drop_duplicates()
    print('Simply cleaned rows:', len(clean_data))

    # define random seed
    seed = 1234
    test_size = 0.2
    np.random.seed(seed)
    labels = clean_data.pop(label_name)

    train_data, test_data, train_labels, test_labels = train_test_split(
        clean_data, labels, test_size=test_size, random_state=seed)

    # run original train set
    run_pipeline((train_data, train_labels), (test_data, test_labels), numerics, categories)
    # generate dirty dataset
    dirty_data = train_data.copy()
    feature_transformation = ColumnTransformer(transformers=[
        ('numerical_features', StandardScaler(), numerics),
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categories),
        ('textual_features', TfidfVectorizer(), 'text'),
    ], remainder="drop")

    outliers, indices = generate_multivariate_outliers(dirty_data, numerics, categories, percentage=3,
                                                       factors=[1 / 2, 2])
    outlier_labels = np.random.choice(['small', 'fit', 'large'], size=len(outliers))
    outlier_labels = DataFrame(outlier_labels)
    dirty_data, transformed_data = merge_outliers(dirty_data, outliers, feature_transformation, visualization=False)
    dirty_labels = pd.concat([outlier_labels, train_labels])

    run_pipeline((dirty_data, dirty_labels), (test_data, test_labels), numerics, categories)

    # add_outliers(dirty_data, numerical_columns=numerics, categorical_columns=categories,
    #              outlier_percentage=25, factor=10)
    # add_noise_to_text(dirty_data, percentage=90, noise_percentage=40)
    # cleaned_data = dirty_data.copy()
    # detect_outliers(cleaned_data, categories)
    # textprocess(cleaned_data, 'text')
    # for column in numerics:
    #     remove_outliers_iqr(cleaned_data, column)

    # clean the dirty data
    clean_start_time = time.time()
    clean_data = dirty_data.reset_index(drop=True)
    clean_labels = dirty_labels.reset_index(drop=True)

    p = len(outliers) / len(dirty_data) * 100
    outlier_pos = detect_multivariate_outliers(transformed_data, percentile=p, visualization=False)
    detected_outliers_num = np.sum(outlier_pos < len(outliers))
    print('Detected generated outliers percent:', detected_outliers_num / len(outliers))

    clean_end_time = time.time()
    cleaned_data = clean_data.drop(index=outlier_pos)
    cleaned_labels = clean_labels.drop(index=outlier_pos)

    run_pipeline((cleaned_data, cleaned_labels), (test_data, test_labels), numerics, categories)
    train_end_time = time.time()
    print(
        f'Cleaning data costs {clean_end_time - clean_start_time}s\n'
        f'Training data costs {train_end_time - clean_end_time}s')
