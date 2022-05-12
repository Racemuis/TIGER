from pathlib import Path
import re
import os
import logging
logging.basicConfig(level = logging.INFO)

import numpy as np
from sklearn.model_selection import train_test_split

from i_o import process_folder

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

def clean_train_test_split(X_directory: Path, y_directory: Path, test_size: float, shuffle: bool = True, random_state: int = None):
    """Returns the filenames that result in a clean train-test split where the intersection between the patient IDs in the train and test set is empty
    Args:
        X (Path): directory containing the data
        y (Path): directory containing the labels
        test_size (int): the size of the test data set (not guaranteed)
        shuffle (bool): the shuffle parameter, Default = True
        random_state (int): the shuffle seed

    Raises:
         AssertionError: raises when sizes of created train and test sets do not match

    Returns:
        np.array[str]: filenames of the train data
        np.array[str]: filenames of the test data
        np.array[str]: filenames of the train labels
        np.array[str]: filenames of the test labels
        np.array[str]: filenames of the train msks
        np.array[str]: filenames of the test msks
    """
    X_filenames = np.array(os.listdir(X_directory))   
    y_filenames = np.array(os.listdir(y_directory))

    X_train_files, X_test_files, y_train_files, y_test_files = train_test_split(X_filenames, y_filenames, test_size=test_size, shuffle = shuffle, random_state = random_state)

    X_train_IDs = np.array([get_ID(f) for f in X_train_files])
    X_test_IDs = np.array([get_ID(f) for f in X_test_files])
    intersect_idx = np.in1d(X_train_IDs, X_test_IDs)

    if np.sum(intersect_idx) > 0:
        logging.info(f"Moving {np.sum(intersect_idx)} item(s) to the test set to separate patient data.")

        X_test_files = np.append(X_test_files, X_train_files[intersect_idx])
        y_test_files = np.append(y_test_files, y_train_files[intersect_idx])

        X_train_files = np.delete(X_train_files, intersect_idx)
        y_train_files = np.delete(y_train_files, intersect_idx)

    msks_train_files = [os.path.splitext(img)[0]+"_tissue"+ os.path.splitext(img)[1] for img in X_train_files]
    msks_test_files = [os.path.splitext(img)[0]+"_tissue"+ os.path.splitext(img)[1] for img in X_test_files]

    assert X_train_files.shape[0] + X_test_files.shape[0] == len(X_filenames), f"Expected the number of samples in the train and test set to add up to {len(X_filenames)}, got {X_train.shape[0] + X_test.shape[0]}."
    assert X_train_files.shape[0] == y_train_files.shape[0], f"Expected X_train ({ X_train_files.shape[0]}) and y_train ({y_train_files.shape[0]}) to have the same length."
    assert X_test_files.shape[0] == y_test_files.shape[0], f"Expected X_test ({ X_test_files.shape[0]}) and y_test ({y_test_files.shape[0]}) to have the same length."

    return X_train_files, X_test_files, y_train_files, y_test_files, msks_train_files, msks_test_files
    
