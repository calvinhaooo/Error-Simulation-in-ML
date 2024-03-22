import pandas as pd
from sim_errors import introduce_missing_values, introduce_outliers
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from simulate_errors import *


electronics_path = './data/df_electronics.csv'
modcloth_path = './data/df_modcloth.csv'

def extract_datetime_features(X):
    features = pd.DataFrame(index=X.index)
    
    features['year'] = X['timestamp'].dt.year
    features['month'] = X['timestamp'].dt.month
    features['day'] = X['timestamp'].dt.day
    features['weekday'] = X['timestamp'].dt.weekday
    features['hour'] = X['timestamp'].dt.hour
    
    return features

def define_training_pipeline(numerical_columns, categorical_columns, timestamp_columns) -> Pipeline:
    print("Setting up training pipeline")
    
    timestamp_transformer = FunctionTransformer(extract_datetime_features, validate=False)
    
    feature_transformation = ColumnTransformer(transformers=[
        ('numerical_features', StandardScaler(), numerical_columns),
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('timestamp_features', timestamp_transformer, timestamp_columns),
    ], remainder="drop")

    pipeline = Pipeline([
        ('features', feature_transformation),
        ('classifier', LogisticRegression(multi_class='multinomial', max_iter=5000))
    ])

    return pipeline

def run_pipeline(train_set, test_set, numerical_columns, categorical_columns, timestamp_columns):
    train_x, train_y = train_set
    test_x, test_y = test_set
    print("---------------------------------------")
    sklearn_pipeline = define_training_pipeline(numerical_columns, categorical_columns, timestamp_columns)
    print("Start training pipeline")
    model = sklearn_pipeline.fit(train_x, train_y)
    pred = model.predict(test_x)
    prob = model.predict_proba(test_x)
    evaluate(y_scores=prob, y_pred=pred, y_true=test_y)


def evaluate(y_scores, y_pred, y_true):
    auc = roc_auc_score(y_true, y_scores, average='macro', multi_class='ovo')  # or average='micro'/'weighted'
    print("AUC Score:", auc)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    precision = precision_score(y_true, y_pred, average='macro', zero_division='warn')  # or average='micro'/'weighted'
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

    electronics, modcloth = load_datasets(electronics_path, modcloth_path)

    label_name = 'fit'

    electronics = preprocess_electronics(electronics)
    modcloth = preprocess_modcloth(modcloth)
    
    common_columns = ['item_id', 'user_id', 'rating', 'timestamp', 'model_attr', 'category', 'brand', 'year', 'user_attr', 'split']

    merged_df = merge_data(modcloth, electronics, common_columns)

    categories, numerics, texts, timestamps = select_features(merged_df, label_name, alpha=0.2)

    print("--------------------------------------------------------------------------------------------")
    print("Categorical columns:", categories)
    print("Numerical columns:", numerics)
    print("Text columns:", texts)
    print("Timestamp columns:", timestamps)

    # Print final columns 
    print("--------------------------------------------------------------------------------------------")
    final_columns = categories + numerics + timestamps + [label_name]
    print("Final columns:", final_columns)

    # Clean the data 
    print("--------------------------------------------------------------------------------------------")
    print("Cleaning the data")
    processed_data = merged_df[final_columns]
    print(len(processed_data))
    clean_data = processed_data.dropna()
    clean_data = clean_data.drop_duplicates()
    print(len(clean_data))

    # Train test split 
    seed = 1234
    test_size = 0.2
    np.random.seed(seed)
    labels = clean_data.pop(label_name)
    
    print("--------------------------------------------------------------------------------------------")
    train_data, test_data, train_labels, test_labels = train_test_split(
        clean_data, labels, test_size=test_size, random_state=seed)
    print("Train data", train_data.shape)
    print("Test data", test_data.shape)
    
    print("--------------------------------------------------------------------------------------------")
    print("Introducing missing values")
    dirty_data = train_data.copy()
    dirty_data = introduce_timestamp_errors(dirty_data)

    # Run pipeline 
    print("--------------------------------------------------------------------------------------------")
    print("Running pipeline without dirty data")
    run_pipeline((train_data, train_labels), (test_data, test_labels), numerics, categories, timestamps)
    print("--------------------------------------------------------------------------------------------")
    print("Running pipeline with dirty data")
    run_pipeline((dirty_data, train_labels), (test_data, test_labels), numerics, categories, timestamps)
   
    # Adding random noise results

    # Adding outliers results 

    # Timestamp error results

