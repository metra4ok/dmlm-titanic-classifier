import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accommodate sklearn pipeline functionality
        return self

    def transform(self, X):
        # add indicator
        X = X.copy()
        for var in self.variables:
            X[var+'_na'] = np.where(X[var].isnull(), 1, 0)
        return X


# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna('Missing')
        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist median in a dictionary
        self.imputer_dict_ = {}
        for var in self.variables:
            self.imputer_dict_[var] = X[var].median()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var].fillna(self.imputer_dict_[var], inplace=True)
        return X


# Extract first letter from string variable
class ExtractFirstLetter(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def _extract_letter(self, row):
        try:
            return row.str()[0]
        except:
            return np.nan

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].apply(self._extract_letter)
        return X


# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.tol = tol

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}
        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.encoder_dict_[var]),
                              X[var], "Rare")
        return X


# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # HINT: persist the dummy variables found in train set
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        # get dummies
        X = pd.concat([X, pd.get_dummies(X[self.variables], drop_first=True)],
                      axis=1)
        # drop original variables
        X.drop(self.variables, axis=1, inplace=True)
        # add missing dummies if any
        missing_dummies = [var for var in self.dummies if var not in X.columns]
        if len(missing_dummies) != 0:
            for var in missing_dummies:
                X[var] = 0
        return X
