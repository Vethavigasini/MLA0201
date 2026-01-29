import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = np.array([
    [2, 1],
    [1, 2],
    [2, 3],
    [3, 3],
    [3, 2],
    [7, 6],
    [8, 7],
    [7, 8],
    [8, 8],
    [9, 8]
], dtype=float)

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

new_sample = np.array([[4, 3]], dtype=float)
pred = model.predict(new_sample)[0]
prob = model.predict_proba(new_sample)[0]

print("New sample:", new_sample[0].tolist())
print("Predicted class:", int(pred))
print("Probability [class0, class1]:", prob.tolist())
