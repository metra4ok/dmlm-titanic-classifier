import pathlib

import titanic_classifier_model


PACKAGE_ROOT = pathlib.Path(titanic_classifier_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"
PIPELINE_FILENAME = "classification_model"

# data
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
TARGET = "survived"
FEATURES = ['pclass', 'sex', 'age', 'sibsp', 'parch', 
            'fare', 'cabin', 'embarked', 'title']

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_VARS = ['age', 'fare']

CABIN = 'cabin'

NUMERICAL_NA_NOT_ALLOWED = ['pclass', 'sibsp', 'parch']
