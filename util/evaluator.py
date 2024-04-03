from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import *
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def define_training_pipeline(numerical_columns, categorical_columns, classifier, vectorizer=None,
                             reduction=0) -> Pipeline:
    if vectorizer is None:
        vectorizer = TfidfVectorizer()

    print("Setting up training pipeline")
    feature_transformation = ColumnTransformer(transformers=[
        ('numerical_features', StandardScaler(), numerical_columns),
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('textual_features', vectorizer, 'text'),
    ], remainder="drop")

    if reduction == 0:
        pipeline = Pipeline([
            ('features', feature_transformation),
            ('classifier', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('features', feature_transformation),
            ('reduction', TruncatedSVD(n_components=reduction)),
            ('classifier', classifier)
        ])

    return pipeline


def run_pipeline(train_set, test_set, numerical_columns, categorical_columns, model, task_type, vectorizer=None,
                 reduction=0):
    train_x, train_y = train_set
    test_x, test_y = test_set
    print("---------------------------------------")
    sklearn_pipeline = define_training_pipeline(numerical_columns, categorical_columns, model, vectorizer, reduction)
    # train the model
    print("Start training pipeline")
    model = sklearn_pipeline.fit(train_x, train_y)
    # evaluate the model on the test set
    if task_type == 'classification':
        pred = model.predict(test_x)
        prob = model.predict_proba(test_x)
        res = evaluate_classifier(y_scores=prob, y_pred=pred, y_true=test_y)
    elif task_type == 'regression':
        pred = model.predict(test_x)
        res = evaluate_regression(y_pred=pred, y_true=test_y)
    else:
        raise ValueError("Unknown task type: {}".format(task_type))
    return res


def evaluate_classifier(y_scores, y_pred, y_true):
    if len(y_scores[0]) == 2:
        y_scores = [scores[1] for scores in y_scores]

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
    return auc, accuracy, precision, recall, f1


def evaluate_regression(y_pred, y_true):
    mse = mean_squared_error(y_true, y_pred)
    print("MSE:", mse)
    rmse = mse ** 0.5
    print("RMSE:", rmse)
    r2 = r2_score(y_true, y_pred)
    print("R² Score:", r2)
    return mse, rmse, r2
