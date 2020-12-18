from flask import Blueprint, request, jsonify
from titanic_classifier_model.predict import make_prediction
from titanic_classifier_model import __version__ as model_version

from api import __version__ as api_version
from api.config import get_logger
from api.validation import validate_inputs

import json


_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status ok')
        return 'ok'


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': model_version,
                        'api_version': api_version})


@prediction_app.route('/v1/predict/classification', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        _logger.info(f'Inputs: {json_data}')

        input_data, errors = validate_inputs(input_data=json_data)
        _logger.info(f'\n!!!\nValidated inputs: {input_data}')

        result = make_prediction(input_data=input_data)
        _logger.info(f'Outputs: {result}')

        predictions = result.get('predictions').tolist()
        version = result.get('version')

        return jsonify({'predictions': predictions,
                        'errors': errors,
                        'version': version})


