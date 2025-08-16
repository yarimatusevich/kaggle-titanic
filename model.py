import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from data import load_data, feature_extract, clean_features

data = load_data()
features = feature_extract(data)
cleaned_features = clean_features(features)

X_train = cleaned_features.X_train
y_train = cleaned_features.y_train
X_test = cleaned_features.X_test
y_test = cleaned_features.y_test

# creating model
model = RandomForestClassifier(n_estimators=100, random_state=1)

# training model
model.fit(X=X_train, y=y_train)

# prediction and model metrics
y_pred = model.predict(X=X_test)

acc = accuracy_score(y_true=y_test, y_pred=y_pred)
prec = precision_score(y_true=y_test, y_pred=y_pred, average='binary')
recall = recall_score(y_true=y_test, y_pred=y_pred, average='binary')

results = pd.DataFrame({
    'PassengerId': data.test['PassengerId'],
    'Survived': y_pred
})

results.to_csv('predictions.csv', index=False)

print(f'acc: {acc}, precision: {prec}, recall: {recall}')