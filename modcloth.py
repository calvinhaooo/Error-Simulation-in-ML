import numpy as np
import pandas as pd
from jenga.basis import Task, MultiClassClassificationTask
from jenga.evaluation.corruption_impact import CorruptionImpactEvaluator

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from data_preprocessor import select_features


def define_training_pipeline(numerical_columns, categorical_columns):
    print("Setting up training pipeline")
    # IMPLEMENT ME
    feature_transformation = ColumnTransformer(transformers=[
        ('numerical_features', StandardScaler(), numerical_columns),
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('textual_features', CountVectorizer(), 'text'),
    ], remainder="drop")

    pipeline = Pipeline([
        ('features', feature_transformation),
        ('learner', SGDClassifier(loss='log_loss', penalty='l1', max_iter=1000, learning_rate="adaptive", eta0=0.01))
        # ('classifier', RandomForestClassifier()) # 又慢又不准
        # ('classifier', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    ])

    # param_grid = {
    #     'learner__n_estimators': [30],
    #     'learner__max_depth': [None, 5, 10],
    # }
    # grid_search = GridSearchCV(pipeline, param_grid, cv=5)

    return pipeline


def run_reviews_pipeline(seed, test_size):
    np.random.seed(seed)
    mc_data = pd.read_json('./data/modcloth_final_data.json', lines=True)

    label_name = 'fit'

    categorical_columns, numerical_columns, text_columns = select_features(label_name, mc_data)
    mc_data['text'] = ''
    for text_column in text_columns:
        mc_data['text'] += ' ' + mc_data[text_column]

    final_columns = categorical_columns + numerical_columns + ['text', label_name]
    data = mc_data[final_columns].dropna()
    print(len(data))
    labels = data.pop('fit')

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=seed)
    sklearn_pipeline = define_training_pipeline(numerical_columns, categorical_columns)
    model_with_transformations = sklearn_pipeline.fit(train_data, train_labels)
    accuracy = model_with_transformations.score(test_data, test_labels)
    print(f"Accuracy on test data is {accuracy}")

    return accuracy


accuracy = run_reviews_pipeline(1234, 0.2)
print(accuracy)
