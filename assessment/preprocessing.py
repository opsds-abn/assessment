"""Module containing helper functions to preprocess data"""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from assessment.constants import PERSONS_OF_INTEREST_DATA_PATH
import numpy as np
from scipy.stats import linregress


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


def get_statistics_description(variable_series: pd.Series, poi_series: pd.Series) -> Tuple:
    """
    Basic statistics of a variable
    :param variable_series: pandas series of the variable in question
    :param poi_series: pandas series containing the poi variable
    :return: a tuple with mean/std of variable for poi/non-poi people, correlation between variable and poi, and the
             percentage of missing values
    """
    # Split variable among poi/not-poi
    variable_poi = variable_series[poi_series]
    variable_no_poi = variable_series[poi_series == False]

    # Get the percentage of missing values in the total variable
    missing_values = sum(np.isnan(variable_series)) / variable_series.__len__() * 100

    # For poi/non-poi, get the average variable value and the std (leaving out nan values)
    variable_poi_mean, variable_poi_std = variable_poi.mean(), variable_poi.std()
    variable_no_poi_mean, variable_no_poi_std = variable_no_poi.mean(), variable_no_poi.std()

    # Calculate the r-value between poi and the target variable (only possible for the non-NAN values)
    is_real = ~np.isnan(variable_series)
    _, _, r, _, _ = linregress(variable_series[is_real], poi_series[is_real])

    return variable_poi_mean, variable_poi_std, variable_no_poi_mean, variable_no_poi_std, r, missing_values


def data_normalization(full_data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Adding in the missing data and then normalizing the variables according to the variable mean mu and standard
    deviation sigma
    x' = (x - mu) / sigma
    :param full_data_frame: contains all the unnormalized data
    :return:
    """
    # First, we do not consider director_fees and loan_advances within this analysis
    full_data_frame = full_data_frame.drop(['director_fees', 'loan_advances', 'restricted_stock_deferred'], axis=1)

    # Next, for the following variables we replace all nan values by the variable median
    for variable in ['salary', 'to_messages', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value',
                     'expenses','from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other',
                     'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock']:
        full_data_frame[variable] = full_data_frame[variable].fillna(full_data_frame[variable].median())

    # The remaining variables we replace all nan values with a zero
    for variable in ['deferral_payments']:
        full_data_frame[variable] = full_data_frame[variable].fillna(0)

    # Now we normalize all the data
    for variable in full_data_frame.columns:
        if variable not in ['poi']:
            full_data_frame[variable] = (full_data_frame[variable] - full_data_frame[variable].mean()) / full_data_frame[variable].std()

    return full_data_frame
