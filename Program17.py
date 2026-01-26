import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("/Users/vethavigasini/Downloads/mobile_prices.csv")

print("Dataset Preview:\n", data.head())

plt.figure(figsize=(10, 8))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm", linewidths=1)
plt.title("Feature Correlation Heatmap")
plt.show()

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

lreg = LogisticRegression(max_iter=200)
lreg.fit(x_train, y_train)

y_pred = lreg.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print("\nAccuracy of Logistic Regression Model: {:.2f}%".format(accuracy))

(unique, counts) = np.unique(y_pred, return_counts=True)
price_range = np.asarray((unique, counts)).T
print("\nPredicted Class Distribution:\n", price_range)

plt.figure(figsize=(10, 5))
plt.plot(y_test[:20], marker='o', label='Actual', linestyle='-', color='blue')
plt.plot(y_pred[:20], marker='x', label='Predicted', linestyle='--', color='red')
plt.title("Actual vs Predicted (First 20 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Price Range")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x=y_pred, palette="Set2")
plt.title("Predicted Price Range Distribution")
plt.xlabel("Price Range Class")
plt.ylabel("Count")
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
