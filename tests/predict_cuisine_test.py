import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np

from project2 import predict_cuisine


def test_predict_cuisine():
    x_train = ["one two three", "four five six", "seven eight nine"]
    y_train = ["cuisine1", "cuisine2", "cuisine3"]
    knn = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=2))
    ])
    knn.fit(x_train, y_train)

    ingredients = ["one four seven"]
    dist, index = predict_cuisine(knn, ingredients)
    assert isinstance(dist, np.ndarray)
    assert isinstance(index, np.ndarray)
    assert len(dist[0]) == len(index[0])
