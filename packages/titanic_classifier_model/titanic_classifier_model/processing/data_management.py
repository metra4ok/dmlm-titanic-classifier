import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from titanic_classifier_model.config import config


def load_data(*, filename: str) -> pd.DataFrame:
    return pd.read_csv(config.DATASET_DIR / filename)


def save_pipeline(*, pipeline_to_persist) -> None:
    save_file_name = config.PIPELINE_FILENAME
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)
    print("pipeline saved")


def load_pipeline(*, pipeline_filename: str) -> Pipeline:
    file_path = config.TRAINED_MODEL_DIR / pipeline_filename
    pipe = joblib.load(filename=file_path)
    return pipe


