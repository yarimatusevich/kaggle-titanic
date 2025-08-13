from data import get_data

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = get_data()

# Features (What model is looking at to make prediction)
X = df[['peak', 'gene', 'Pair']]

# Target (True/False)
y = df['Peak2Gene']

# The split shufflers data around 'random_state=42' means we are basically setting a seed for the random shuffle
# As long as we decalre random_state our accuracy will be consistent because the same data will always be used for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2004
)

model = RandomForestClassifier(random_state=2004)
model.fit(X_train, y_train)

y_predictions = model.predict(X_test)
print(classification_report(y_true=y_test,y_pred=y_predictions))

"""
common_cells = rna.columns.intersection(atac.columns)
"""