from utils.visualize import extract_features

def cluster(pca, kmeans, model, dataset, batch_size, return_features=False, learn="predict"):
    features = extract_features(model, dataset, batch_size)
    reduced = pca.fit_transform(features)
    if learn=="predict":
        pseudo_labels = list(kmeans.predict(reduced))
    elif learn=="fit_predict":
        pseudo_labels = list(kmeans.fit_predict(reduced))
    else:
        pseudo_labels = list(kmeans.fit(reduced))
    if return_features:
        return pseudo_labels, features
    return pseudo_labels
