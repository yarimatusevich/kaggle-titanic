from dataclasses import dataclass
import pandas as pd
import numpy as np
from pandas import DataFrame

@dataclass
class TitanicData():
    train: DataFrame
    test: DataFrame
    submissions: DataFrame

@dataclass
class Features():
    X_train: DataFrame
    y_train: DataFrame
    X_test: DataFrame
    y_test: DataFrame

def load_data() -> TitanicData:
    return TitanicData(
        train=pd.read_csv('titanic/train.csv'),
        test=pd.read_csv('titanic/test.csv'),
        submissions=pd.read_csv('titanic/gender_submission.csv')
    )

def feature_extract(data: TitanicData) -> Features:
    train = data.train
    test = data.test
    submissions = data.submissions

    return Features(
        X_train = train.drop(columns=['Survived','PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch', 'Fare']),
        y_train = train['Survived'],
        X_test = test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch', 'Fare']),
        y_test = submissions['Survived']
    )

def clean_features(data: Features) -> Features:
    # filling empty fields in age with median
    age_median = data.X_train['Age'].median()
    data.X_train['Age'] = data.X_train['Age'].fillna(age_median)
    data.X_test['Age'] = data.X_test['Age'].fillna(age_median)

    # encoding categorical variabler
    data.X_train['Sex'] = data.X_train['Sex'].map({'male': 0, 'female': 1})
    data.X_test['Sex'] = data.X_test['Sex'].map({'male': 0, 'female': 1})

    data = perform_normalization(data)

    return data

def perform_normalization(data: Features) -> Features:
    # normalizing X_train
    data.X_train['Sex'] = np.log(data.X_train['Sex'] + 1)
    data.X_train['Pclass'] = np.log(data.X_train['Pclass'])
    data.X_train['Age'] = np.log(data.X_train['Age'])

    # # noramlizing y_train
    # data.y_train['Sex'] = np.log(data.y_train['Sex'] + 1)
    # data.y_train['Pclass'] = np.log(data.y_train['Pclass'])
    # data.y_train['Age'] = np.log(data.y_train['Age'])

    # X_test
    data.X_test['Sex'] = np.log(data.X_test['Sex'] + 1)
    data.X_test['Pclass'] = np.log(data.X_test['Pclass'])
    data.X_test['Age'] = np.log(data.X_test['Age'])

    # # y_test
    # data.y_test['Sex'] = np.log(data.y_test['Sex'] + 1)
    # data.y_test['Pclass'] = np.log(data.y_test['Pclass'])
    # data.y_test['Age'] = np.log(data.y_test['Age'])

    return data