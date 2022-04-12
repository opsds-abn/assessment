"""Module containing all the constants and data paths"""

from pathlib import Path

DATA_PATH = Path(__file__).parents[1].joinpath("data")
PERSONS_OF_INTEREST_DATA_PATH = DATA_PATH.joinpath("persons_of_interest.csv")
