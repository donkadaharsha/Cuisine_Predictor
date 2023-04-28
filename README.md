## HARSHA VARDHAN (Donkada.H.Vardhan-1@ou.edu)

## Description
The aim of this project is to develop an application that can receive a list of ingredients from the user and provide predictions on the type of cuisine and suggest similar meals. The project follows a set of steps, which are as - Training (or indexing for search) the food data that is provided, Requesting the user to input the ingredients they are interested in through command-line arguments, Utilizing the model (or search index) to forecast the type of cuisine and conveying the results to the user and Identifying the top-N closest foods (with N defined by a command-line parameter) and sharing the dish IDs with the user.

## Command to run the project

pipenv run python project2.py --N 5 --ingredient paprika --ingredient banana --ingredient "rice krispies"

## Command to run pytest 
pipenv run python -m pytest

## Modules to install
numpy: pipenv install numpy
pandas: pipenv install pandas
sklearn: pipenv install sklearn - TfidfVectorizer, KNeighborsClassifier, CountVectorizer, TfidfTransformer
numpyencoder: pipenv install numpyencoder

## Functions Used

read(json_file): Reads a JSON file - yummly.json and returns its content as a pandas dataframe 

clean_data(input_data): This function takes a Pandas DataFrame as an input and cleans the 'ingredients' column by joining all elements into a single string separated by commas.

train_knn_model(x_train_data, y_train_data): Trains a K-Nearest Neighbors (KNN) model to classify text data. It takes in two arguments, x_train_data which is a list of strings containing the training data, and y_train_data which is a list of the corresponding labels. Inside the function, the CountVectorizer is used to convert the text data into a matrix of token counts, x_train_vect. Then, the TfidfTransformer is used to transform the token count matrix into a matrix of normalized term frequency-inverse document frequency (TF-IDF) values, x_train_tfidf. KNeighborsClassifier is created with n_neighbors set to 20 and is fit on the TF-IDF transformed data and the corresponding labels. Finally, a Pipeline is created with three steps: CountVectorizer, TfidfTransformer, and KNeighborsClassifier. This pipeline is then fit on the original text data and corresponding labels. The trained pipeline is returned at the end of the function.

predict_cuisine(knn_model, ingredients_data): Takes the trained KNN model and a list of ingredients as input, and returns the distances and indices of the k-nearest neighbors for the given ingredients.

def create_result(dist_data, index_data, y_train_data, n_data): This function creates a dictionary result with keys "Cuisine", "score", and "closest", and returns it. The value of "Cuisine" is the value of the "cuisine" column in the first row of the ing_df_subset DataFrame, the value of "score" is the value of the "dist" column in the first row of the ing_df_subset DataFrame rounded to three decimal places, and the value of "closest" is a list of dictionaries with keys "id" and "score", where each dictionary corresponds to one of the closest dishes selected earlier.

def result(ingredients, n, input_data): The function first extracts the ingredients and cuisine columns from the input_data DataFrame and then splits the data into training and testing sets using train_test_split from sklearn.model_selection. Then the function trains a K-Nearest Neighbors (KNN) model on the training data using train_knn_model function. After that, the predict_cuisine function is used to predict the type of cuisine based on the ingredients list passed to the function. The create_result function is then called to create a dictionary containing the n closest food items and their scores, as well as the predicted cuisine and its score. Finally, the function prints the result in a JSON format and returns the result dictionary.

Sample output:

{
    "Cuisine": "spanish", 
    "score": 0.979, 
    "closest": [
        {
            "id": 8063, 
            "score": 1.078
        }, 
        {
            "id": 12629, 
            "score": 1.079
        }, 
        {
            "id": 20713, 
            "score": 1.083
        }

## Test cases

read_test: The function first creates a fixture input_data which is a pandas DataFrame containing some ingredients and cuisine values. Then, it uses the tmp_path fixture provided by pytest to create a temporary directory and saves the input data to a JSON file in that directory. Then it calls the read() function with the path to the JSON file and saves the result to the result variable. Finally, it uses an assert statement to check if the result dataframe is equal to the input_data dataframe. If they are equal, the test passes.

clean_data_test : This creates a simple DataFrame with two rows of ingredients data. The function then calls clean_data with the input data, and uses the assert statements to check if the resulting 'ingredients' column of the cleaned data frame matches the expected comma-separated strings. If the assertions pass, the test is considered successful.

predict_cuisine_test: predict_cuisine function is called with a single test case consisting of a list of ingredients. The function should return two values: a distance array and an index array. The assert statements at the end of the function check that both of these values are NumPy arrays and that they have the same length. If all assertions pass, the test case is considered to have passed. If any assertion fails, the test case is considered to have failed.

## Assumptions and Bugs

- It is possible that errors may occur if new ingredients are given to the program.

- It is assumed that the similarity between the ingredients of the user-provided food and the ingredients of the N most similar dishes is represented by their associated scores.

- Provided that the ingredients entered via the command line interface (CLI) are present in the yummly.json file.

## Video



https://user-images.githubusercontent.com/114453047/235035503-cb132adf-35a2-4a80-b4c9-acbba30cedfa.mp4



