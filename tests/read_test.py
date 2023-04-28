import pandas as pd
import pytest

from project2 import read

@pytest.fixture
def input_data():
    return pd.DataFrame({'ingredients': ['onions', 'garlic'], 'cuisine': ['italian', 'mexican']})

def test_read(input_data, tmp_path):
    # Save input data to a JSON file in a temporary directory
    json_file = tmp_path / 'input_data.json'
    input_data.to_json(json_file)
    
    # Call the function with the path to the JSON file
    result = read(json_file)
    
    # Assert that the result is the same as the input data
    assert result.equals(input_data)
