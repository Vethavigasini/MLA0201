import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Importing the dataset
data = pd.read_csv("CarPrice.csv")

# Data Exploration
print(data.head())
print("Shape:", data.shape)
print("Missing values:\n", data.isnull().sum())
print(data.info())
print(data.describe())
print("Unique car names:", data['CarName'].unique())

# Analyzing correlations with only numeric data
numeric_data = data.select_dtypes(include='number')
print(numeric_data.corr())

plt.figure(figsize=(20, 15))
sns.heatmap(numeric_data.corr(), cmap="coolwarm", annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Training a Car Price Prediction Model
predict = "price"
features = ["symboling", "wheelbase", "carlength", 
            "carwidth", "carheight", "curbweight", 
            "enginesize", "boreratio", "stroke", 
            "compressionratio", "horsepower", "peakrpm", 
            "citympg", "highwaympg"]

# Extracting features and target
x = data[features].values
y = data[predict].values

# Splitting data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Decision Tree Regressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)

# Predictions
predictions = model.predict(xtest)

# Evaluation
mae = mean_absolute_error(ytest, predictions)
score = model.score(xtest, ytest)

print(f"\nModel Score (R²): {score:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")
