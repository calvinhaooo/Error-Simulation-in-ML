import time
import warnings

from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from util.data_cleaner import *
from util.data_preprocessor import *
from util.error_generator import *
from util.evaluator import *

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DataConversionWarning)

    dataframe = read_data(file_name='modcloth_final_data.json')
    label_name = 'fit'

    dataframe['cup size'] = dataframe['cup size'].apply(convert_cup_size_to_cms)
    dataframe['height'] = dataframe['height'].apply(convert_height_to_number)

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
    task = 'classification'
    model = SGDClassifier(loss='log_loss', random_state=seed)

    # run original train set
    train_data, test_data, train_labels, test_labels = train_test_split(
        clean_data, labels, test_size=test_size, random_state=seed)

    run_pipeline((train_data, train_labels), (test_data, test_labels), numerics, categories, model, task)
    # generate dirty dataset
    dirty_data = clean_data.copy()
    dirty_data[label_name] = labels
    cleaned_data = dirty_data.copy()

    random_replace_column(dirty_data, label_name, 500)  # alert num_errors here
    labels = dirty_data.pop(label_name)

    train_data, test_data, train_labels, test_labels = train_test_split(
        dirty_data, labels, test_size=test_size, random_state=seed)
    print(train_labels.value_counts())
    print(test_labels.value_counts())
    run_pipeline((train_data, train_labels), (test_data, test_labels), numerics, categories, model, task)

    # clean the dirty data
    clean_start_time = time.time()

    cleaned_data = cluster_labels(cleaned_data, label_name)
    labels = cleaned_data.pop(label_name)

    clean_end_time = time.time()
    train_data, test_data, train_labels, test_labels = train_test_split(
        cleaned_data, labels, test_size=test_size, random_state=seed)

    run_pipeline((train_data, train_labels), (test_data, test_labels), numerics, categories, model, task)

    train_end_time = time.time()
    print(
        f'Cleaning data costs {clean_end_time - clean_start_time}s\n'
        f'Training data costs {train_end_time - clean_end_time}s')
