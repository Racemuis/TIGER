from pathlib import Path
import re

import numpy as np
from sklearn.model_selection import train_test_split

def get_ID(filename: str)-> str:
    """extracts the patient ID from the filename

    Args:
        filename (str): filename

    Returns:
        str: patient ID, None if not existent
    """
    RUMC_regex = r'^TC_S01_P000(\d+)_C0001_B\d+'
    Institut_Jules_Bordet_regex = r'^(\d+)[BS]'
    TCGA_regex = r'^TCGA-\w+-(\w+)-*'

    for regex in [RUMC_regex, Institut_Jules_Bordet_regex, TCGA_regex]:
        match = re.search(regex, filename)

        if match:
            ID = match.group(1)
            return ID

        
    print(f"The patient ID could not be matched from: {filename}; returning None.")
    return None

def clean_train_test_split(X: np.array, y: np.array, test_size: int, shuffle: bool = True, random_state: int = None):
    """Creates a clean train-test split where the intersection between the patient IDs in the train and test set is empty
    Args:
        X (np.array): data
        y (np.array): labels
        test_size (int): the size of the test data set (not guaranteed)
        shuffle (bool): the shuffle parameter, Default = True
        random_state (int): the shuffle seed

    Returns:
        np.array: train data
        np.array: train labels
        np.array: test data
        np.array: test labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle = shuffle, random_state = random_state)
    # TODO: Implement some object that allows for transferring the name and the array in one go to shuffle the images. 
    
    
