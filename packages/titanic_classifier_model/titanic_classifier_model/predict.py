import pandas as pd

import joblib

from titanic_classifier_model.config import config
from titanic_classifier_model.processing.data_management import load_pipeline
from titanic_classifier_model.processing.validation import validate_inputs
from titanic_classifier_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)

_titanic_pipe = load_pipeline(pipeline_filename=f"{config.PIPELINE_FILENAME}_v{_version}.pkl")


def make_prediction(*, input_data) -> dict:
    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(data)
    prediction = _titanic_pipe.predict(validated_data[config.FEATURES])
    response = {'predictions': prediction, 'version': _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {validated_data} "
        f"Predictions: {prediction}"
    )

    return response
   
