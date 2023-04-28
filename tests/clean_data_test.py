import pandas as pd
import pytest

from project2 import clean_data

@pytest.fixture
def input_data():
    return pd.DataFrame({'ingredients': [['salt', 'pepper'], ['sugar']]})

def test_clean_data(input_data):
    cleaned_data = clean_data(input_data)
    assert cleaned_data['ingredients'][0] == 'salt,pepper'
    assert cleaned_data['ingredients'][1] == 'sugar'
