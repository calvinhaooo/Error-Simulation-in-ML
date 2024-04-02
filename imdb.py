from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

from data_cleaner import *
from data_preprocessor import *
from error_generator import *
from evaluator import *


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
    model = BernoulliNB()

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

    run_pipeline((train_data, train_labels), (test_data, test_labels), numerics, categories, model, task_type)
    run_pipeline((train_data, train_labels_modified), (test_data, test_labels), numerics, categories, model, task_type)
    run_pipeline((train_data, train_labels_clean_str), (test_data, test_labels), numerics, categories, model, task_type)
