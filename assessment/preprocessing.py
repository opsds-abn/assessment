"""Module containing helper functions to preprocess data"""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from assessment.constants import PERSONS_OF_INTEREST_DATA_PATH


def read_persons_of_interest() -> pd.DataFrame:
    """Helper function to read in persons of interest data
    from data path and do basic cleaning

    Returns:
        pd.DataFrame: Persons of interest data
    """
    persons_of_interest = pd.read_csv(PERSONS_OF_INTEREST_DATA_PATH)
    persons_of_interest = persons_of_interest.set_index("person_id")

    return persons_of_interest


def get_train_test_split(persons_of_interest: pd.DataFrame) -> Tuple:
    """Helper function to split the  data

    Args:
        persons_of_interest (pd.DataFrame): Persons of interests

    Returns:
        Tuple: Containing the data X_train, X_test, y_train, y_test
    """

    target_column = "poi"
    target = persons_of_interest[target_column]
    features = persons_of_interest.drop(columns=[target_column])

    split = train_test_split(features, target, test_size=0.15, stratify=target)
    return split
