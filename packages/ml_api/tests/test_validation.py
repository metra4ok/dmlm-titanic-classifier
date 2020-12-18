import json

from titanic_classifier_model.config import config as model_config
from titanic_classifier_model.processing.data_management import load_data


def test_prediction_endpoint_validation_200(flask_test_client):
    # Given
    # Load the test data from the regression_model package.
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    test_data = load_data(filename=model_config.TESTING_DATA_FILE)
    post_json = test_data.to_json(orient='records')

    # When
    response = flask_test_client.post('/v1/predict/classification',
                                      json=json.loads(post_json))

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    pred_len = len(response_json.get('predictions'))
    err_len = 0
    print(response_json.get('errors'))
    if response_json.get('errors') != None:
        err_len = len(response_json.get('errors'))
    assert pred_len + err_len == len(test_data)


