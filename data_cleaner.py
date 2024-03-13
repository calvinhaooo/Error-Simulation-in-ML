from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def remove_duplication(df, method='simple'):
    new_df = df.copy()
    if method == 'simple':
        new_df = new_df.drop_duplicates()
    elif method == 'similarity':
        pass
        # todo
    return new_df


def cluster_labels(df, label):
    features = df['text']
    labels = df[label]

    # Encoding labels if they are categorical
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    text_vectorizer = TfidfVectorizer()
    encoded_text = text_vectorizer.fit_transform(features)

    # use TruncatedSVD to extract main features
    n_components = 50
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_text = svd.fit_transform(encoded_text)

    # Creating k-NN classifier without specifying the number of neighbors
    knn = KNeighborsClassifier()
    knn.fit(reduced_text, encoded_labels)
    new_labels = knn.predict(reduced_text)
    print(f"Number of label types after clustering is {len(set(new_labels))}.")

    new_df = df.copy()
    new_df[label] = new_labels
    return new_df
