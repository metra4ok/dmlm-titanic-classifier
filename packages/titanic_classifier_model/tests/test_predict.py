from titanic_classifier_model.config import config
from titanic_classifier_model.predict import make_prediction
from titanic_classifier_model.processing.data_management import load_data


def test_make_single_prediction():
    # Given
    test_data = load_data(filename=config.TESTING_DATA_FILE)
    single_test_json = test_data[0:1].to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    assert subject.get('predictions')[0] == 0


def test_make_multiple_predictions():
    # Given
    test_data = load_data(filename=config.TESTING_DATA_FILE)
    original_data_length = len(test_data)
    multiple_test_json = test_data.to_json(orient='records')

    # When
    subject = make_prediction(input_data=multiple_test_json)

    # Then
    assert subject is not None
    assert len(subject.get('predictions')) == 262


