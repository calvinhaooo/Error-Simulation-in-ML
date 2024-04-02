import time
import warnings

from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from data_cleaner import *
from data_preprocessor import *
from error_generator import *
from evaluator import *

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DataConversionWarning)

    dataframe = read_data(file_name='renttherunway_final_data.json')

    label_name = 'fit'

    parse_date(dataframe, 'review_date')
    dataframe['weight'] = pd.to_numeric(dataframe['weight'].str.replace('lbs', '').astype(float))
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

    train_data, test_data, train_labels, test_labels = train_test_split(
        clean_data, labels, test_size=test_size, random_state=seed)

    # run original train set
    run_pipeline((train_data, train_labels), (test_data, test_labels), numerics, categories, model, task)
    # generate dirty dataset
    dirty_data = train_data.copy()
    feature_transformation = ColumnTransformer(transformers=[
        ('numerical_features', StandardScaler(), numerics),
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categories),
        ('textual_features', TfidfVectorizer(), 'text'),
    ], remainder="drop")
    # here to alert parameters!!!
    outliers = generate_multivariate_outliers(dirty_data, numerics, categories, percentage=3,
                                              factors=[1 / 2, 2])
    outlier_labels = np.random.choice(['small', 'fit', 'large'], size=len(outliers))
    outlier_labels = DataFrame(outlier_labels)
    dirty_data, transformed_data = merge_outliers(dirty_data, outliers, feature_transformation, visualization=True)
    dirty_labels = pd.concat([outlier_labels, train_labels])

    run_pipeline((dirty_data, dirty_labels), (test_data, test_labels), numerics, categories, model, task)

    # clean the dirty data
    clean_start_time = time.time()
    clean_data = dirty_data.reset_index(drop=True)
    clean_labels = dirty_labels.reset_index(drop=True)

    p = len(outliers) / len(dirty_data) * 100
    outlier_pos = detect_multivariate_outliers(transformed_data, percentile=p, visualization=True)
    detected_outliers_num = np.sum(outlier_pos < len(outliers))
    print('Detected generated outliers percent:', detected_outliers_num / len(outliers))

    clean_end_time = time.time()
    cleaned_data = clean_data.drop(index=outlier_pos)
    cleaned_labels = clean_labels.drop(index=outlier_pos)

    run_pipeline((cleaned_data, cleaned_labels), (test_data, test_labels), numerics, categories, model, task)
    train_end_time = time.time()
    print(
        f'Cleaning data costs {clean_end_time - clean_start_time}s\n'
        f'Training data costs {train_end_time - clean_end_time}s')
