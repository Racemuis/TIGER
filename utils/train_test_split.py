from pathlib import Path
import re
import os
import logging
logging.basicConfig(level = logging.INFO)

import numpy as np
from sklearn.model_selection import train_test_split

from io import process_folder

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

def clean_train_test_split(X_directory: Path, y_directory: Path, test_size: int, shuffle: bool = True, random_state: int = None):
    """Creates a clean train-test split where the intersection between the patient IDs in the train and test set is empty
    Args:
        X (Path): directory containing the data
        y (Path): directory containing the labels
        test_size (int): the size of the test data set (not guaranteed)
        shuffle (bool): the shuffle parameter, Default = True
        random_state (int): the shuffle seed

    Raises:
         AssertionError: raises when sizes of created train and test sets do not match

    Returns:
        np.array: train data
        np.array: train labels
        np.array: test data
        np.array: test labels
    """
    X_filenames = os.listdir(X_directory)   
    y = process_folder(y_directory)

    X_train_files, X_test_files, y_train, y_test = train_test_split(X_filenames, y, test_size=test_size, shuffle = shuffle, random_state = random_state)

    X_train_IDs = np.array([get_ID(f) for f in X_train_files])
    X_test_IDs = np.array([get_ID(f) for f in X_test_files])
    intersect_idx = np.in1d(X_train_IDs, X_test_IDs)

    if np.sum(intersect_idx) > 0:
        logging.info(f"Moving {np.sum(intersect_idx)} item(s) to the test set to separate patient data.")

    X_test_files = np.append(X_test_files, X_train_files[intersect_idx])
    y_test = np.append(y_test, y_train[intersect_idx])

    X_train_files = np.delete(X_train_files, intersect_idx)
    y_train = np.delete(y_train, intersect_idx)
    
    X_train = process_folder(X_directory, X_train_files)
    X_test = process_folder(X_directory, X_test_files)

    assert X_train.shape[0] + X_test.shape[0] == len(X_filenames), f"Expected the number of samples in the train and test set to add up to {len(X_filenames)}, got {X_train.shape[0] + X_test.shape[0]}."
    assert X_train.shape[0] == y_train.shape[0], f"Expected X_train ({ X_train.shape[0]}) and y_train ({y_train.shape[0]}) to have the same length."
    assert X_test.shape[0] == y_test.shape[0], f"Expected X_test ({ X_test.shape[0]}) and y_test ({y_test.shape[0]}) to have the same length."

    return X_train, X_test, y_train, y_test
    
