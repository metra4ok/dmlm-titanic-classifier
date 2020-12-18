from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from titanic_classifier_model.processing import preprocessors as pp
from titanic_classifier_model.config import config


titanic_pipe = Pipeline(
        [
            ('extract_first_letter',
             pp.ExtractFirstLetter(variables=config.CABIN)),

            ('numerical_missing_indicator',
             pp.MissingIndicator(variables=config.NUMERICAL_VARS)),

            ('impute_numerical_na_values',
             pp.NumericalImputer(variables=config.NUMERICAL_VARS)),

            ('impute_categorical_na_values',
             pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),

            ('remove_rare_cat_labels',
             pp.RareLabelCategoricalEncoder(tol=0.05, variables=config.CATEGORICAL_VARS)),

            ('one_hot_encoding',
             pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),

            ('scaler',
             StandardScaler()),

            ('logistic_regression',
             LogisticRegression(C=0.0005, random_state=0))
        ]
    )