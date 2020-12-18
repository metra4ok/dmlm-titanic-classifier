import pandas as pd

import joblib

from titanic_classifier_model.config import config
from titanic_classifier_model.processing.data_management import load_pipeline
from titanic_classifier_model.processing.validation import validate_inputs


_titanic_pipe = load_pipeline(pipeline_filename=config.PIPELINE_FILENAME)


def make_prediction(*, input_data) -> dict:
    data = pd.read_json(input_data)
    validated_data = validate_inputs(data)
    prediction = _titanic_pipe.predict(validated_data[config.FEATURES])
    response = {'predictions': prediction}
    return response
   
