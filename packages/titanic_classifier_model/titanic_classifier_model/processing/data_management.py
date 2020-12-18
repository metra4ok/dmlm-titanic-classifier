import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from titanic_classifier_model.config import config
from titanic_classifier_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)

def load_data(*, filename: str) -> pd.DataFrame:
    return pd.read_csv(config.DATASET_DIR / filename)


def save_pipeline(*, pipeline_to_persist) -> None:
    save_file_name = f"{config.PIPELINE_FILENAME}_v{_version}.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    remove_old_pipelines(files_to_keep=save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")


def load_pipeline(*, pipeline_filename: str) -> Pipeline:
    file_path = config.TRAINED_MODEL_DIR / pipeline_filename
    pipe = joblib.load(filename=file_path)
    return pipe


def remove_old_pipelines(*, files_to_keep) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """

    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [files_to_keep, "__init__.py"]:
            model_file.unlink()

