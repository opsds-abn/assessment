"""Module containing helper functions to preprocess data"""

import pandas as pd
from enron.constants import PERSONS_OF_INTEREST_DATA_PATH


def read_persons_of_interest() -> pd.DataFrame:
    """Helper function to read in persons of interest data
    from data path and do basic cleaning

    Returns:
        pd.DataFrame: Persons of interest data
    """
    persons_of_interest = pd.read_csv(PERSONS_OF_INTEREST_DATA_PATH)
    persons_of_interest = persons_of_interest.rename(columns={"Unnamed: 0": "name"})
    persons_of_interest = persons_of_interest.set_index("name")
    return persons_of_interest
