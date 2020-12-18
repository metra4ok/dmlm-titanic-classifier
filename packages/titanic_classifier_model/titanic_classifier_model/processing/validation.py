import pandas as pd

from titanic_classifier_model.config import config


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    validated_data = input_data.copy()

    # check for numerical variables with NA not seen during training
    if input_data[config.NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.NUMERICAL_NA_NOT_ALLOWED
        )

    return validated_data


