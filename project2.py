import argparse
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpyencoder import NumpyEncoder

def read(json_file):
    """
    Reads a JSON file and returns its content as a pandas dataframe.

    Parameters:
    json_file (str): The name or path of the JSON file to be read.

    Returns:
    pandas dataframe: A dataframe containing the content of the JSON file.
    """
    input_data=pd.read_json(json_file)  # read the JSON file using pandas read_json method
    return input_data  # return the dataframe containing the content of the JSON file

# This function takes a Pandas DataFrame as an input 
# cleans the 'ingredients' column by joining all elements into a single string separated by commas. 

def clean_data(input_data):
    input_data['ingredients'] = input_data['ingredients'].apply(lambda x: ','.join(map(str, x)))
    return input_data

def train_knn_model(x_train_data, y_train_data):
    vectorizer = CountVectorizer()     # create a CountVectorizer object
    x_train_vect = vectorizer.fit_transform(x_train_data)  # use the CountVectorizer object to transform the training data into a bag-of-words matrix

    tfidf_transformer = TfidfTransformer()     # create a TfidfTransformer object
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_vect)     # use the TfidfTransformer object to transform the bag-of-words matrix into a TF-IDF matrix
    
    knn_classifier = KNeighborsClassifier(n_neighbors=20)  # create a KNeighborsClassifier object with k=20
    knn_classifier.fit(x_train_tfidf, y_train_data)   # train the KNN classifier on the TF-IDF matrix and the target labels

    # create a Pipeline object consisting of a CountVectorizer, a TfidfTransformer, and a KNeighborsClassifier with k=20
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=20))
    ])
    # train the pipeline on the original training data and the target labels
    pipeline.fit(x_train_data, y_train_data)
        # return the trained pipeline object
    return pipeline


def predict_cuisine(knn_model, ingredients_data):
    # transform the input ingredients into a bag-of-words matrix using the CountVectorizer from the pipeline
    vector = knn_model["vect"].transform(ingredients_data)
    # transform the bag-of-words matrix into a TF-IDF matrix using the TfidfTransformer from the pipeline
    vector = knn_model["tfidf"].transform(vector)
    # convert the sparse matrix to a dense matrix for use with the KNN classifier
    vector = vector.todense()
    # get the distances and indices of the k nearest neighbors in the KNN classifier using the dense vector
    dist, index = knn_model["clf"].kneighbors(np.asarray(vector))
    # return the distances and indices of the k nearest neighbors
    return dist, index

def create_result(dist_data, index_data, y_train_data, n_data):
    # create a DataFrame of the indices, cuisines, and distances of the k nearest neighbors
    ing_df = pd.DataFrame({"id": index_data[0], "cuisine": y_train_data.iloc[index_data[0]].tolist(), "dist": dist_data[0]})
    # subset the DataFrame to only include rows with the same cuisine as the input cuisine
    ing_df_subset = ing_df.loc[ing_df.cuisine == ing_df.cuisine[0]]
    # get the IDs and distances of the k closest neighbors
    closest_ids = ing_df_subset.id.iloc[1:int(n_data)+1] if int(n_data) <= ing_df_subset.shape[0] else ing_df_subset.id.iloc[1:]
    closest_scores = ing_df_subset.dist.iloc[1:int(n_data)+1] if int(n_data) <= ing_df_subset.shape[0] else ing_df_subset.dist.iloc[1:]
    # create a dictionary with the input cuisine, score, and k closest neighbors
    result = {"Cuisine": ing_df_subset.cuisine.iloc[0], "score": round(ing_df_subset.dist.iloc[0], 3), "closest": []}
    # add each closest neighbor to the "closest" list in the result dictionary
    for i in range(len(closest_ids)):
        result["closest"].append({"id": closest_ids.iloc[i], "score": round(closest_scores.iloc[i], 3)})
    # return the result dictionary
    return result


def result(ingredients, n, input_data):
    x = input_data.ingredients
    y = input_data.cuisine
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=42)
    knn = train_knn_model(x_train, y_train)
    dist, index = predict_cuisine(knn, ingredients)
    result = create_result(dist, index, y_train, n)
# Convert result dictionary to JSON and print it
    temp = json.dumps(result, indent=4, sort_keys=False,
                      separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)
    print(temp)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, help="Number of closest meals")
    parser.add_argument("--ingredient", type=str, required=True, help="Ingredients", action='append')
    args = parser.parse_args()
    if args.ingredient and args.N:
        input_data = read(json_file = "yummly.json")
        ingredients = ",".join(args.ingredient)
        input_data = clean_data(input_data)
        result = result([ingredients], args.N, input_data)
