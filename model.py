import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# loading data
train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')
submissions = pd.read_csv('titanic/gender_submission.csv')

# seperating features and targets
X_train = train.drop(columns=['Survived','PassengerId', 'Name', 'Ticket', 'Cabin'])
y_train = train['Survived']
X_test = test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
y_test = submissions['Survived']
ids = test['PassengerId']

# cleaning up age column
age_median = train['Age'].median()
train['Age'].fillna(age_median, inplace=True)
test['Age'].fillna(age_median, inplace=True)

# encoding categorial variables
sex_dict = {'male': 0, 'female': 1}
embarked_dict = {'C': 0, 'S': 1, 'Q': 2}

X_train['Sex'] = X_train['Sex'].map(sex_dict)
X_train['Embarked'] = X_train['Embarked'].map(embarked_dict)

X_test['Sex'] = X_test['Sex'].map(sex_dict)
X_test['Embarked'] = X_test['Embarked'].map(embarked_dict)

# creating model
model = RandomForestClassifier(n_estimators=3000, random_state=2004)

# training model
model.fit(X=X_train, y=y_train)

# prediction and model metrics
y_pred = model.predict(X=X_test)

acc = accuracy_score(y_true=y_test, y_pred=y_pred)
prec = precision_score(y_true=y_test, y_pred=y_pred, average='binary')
recall = recall_score(y_true=y_test, y_pred=y_pred, average='binary')

results = pd.DataFrame({
    'PassengerId': ids,
    'Prediction': y_pred
})

results.to_csv('predictions.csv', index=False)

print(f'acc: {acc}, prev: {prec}, recall: {recall}')