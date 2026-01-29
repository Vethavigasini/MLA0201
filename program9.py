import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
X = np.linspace(-5, 5, 120).reshape(-1, 1)
y = (X[:, 0] ** 2) + np.random.normal(0, 2.0, size=X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)

lin_rmse = mean_squared_error(y_test, lin_pred) ** 0.5
lin_r2 = r2_score(y_test, lin_pred)

# Polynomial Regression
poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("lin", LinearRegression())
])

poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)

poly_rmse = mean_squared_error(y_test, poly_pred) ** 0.5
poly_r2 = r2_score(y_test, poly_pred)

print("Linear Regression")
print("RMSE:", round(lin_rmse, 4))
print("R2 Score:", round(lin_r2, 4))
print()

print("Polynomial Regression")
print("RMSE:", round(poly_rmse, 4))
print("R2 Score:", round(poly_r2, 4))
