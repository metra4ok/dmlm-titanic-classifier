"""The script for downloading and preprocessing datasets

This script were provided for me in the course so I decided 
not to refactor this functionality for now.
"""

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan
    


def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'


data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')

data = data.replace('?', np.nan)

data['cabin'] = data['cabin'].apply(get_first_cabin)
 
data['title'] = data['name'].apply(get_title)

data['fare'] = data['fare'].astype('float')
data['age'] = data['age'].astype('float')

data.drop(labels=['name','ticket', 'boat', 'body','home.dest'], axis=1, inplace=True)

data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)

data_train.to_csv('train.csv', index=False)
data_test.to_csv('test.csv', index=False)