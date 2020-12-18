import pandas as pd

import joblib

from titanic_classifier_model.config import config
from titanic_classifier_model.processing.data_management import load_pipeline


_titanic_pipe = load_pipeline(pipeline_filename=config.PIPELINE_FILENAME)


def make_prediction(*, input_data) -> dict:
    data = pd.read_json(input_data)
    prediction = _titanic_pipe.predict(data[config.FEATURES])
    response = {'prediction': prediction}
    return response
   
