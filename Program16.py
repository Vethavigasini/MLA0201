import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv("/Users/vethavigasini/Downloads/IRIS.csv")
print(iris.head())

x = iris.drop("species", axis=1)
y = iris["species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

decisiontree = DecisionTreeClassifier()
logisticregression = LogisticRegression(max_iter=200)
knearestclassifier = KNeighborsClassifier()
passiveAggressive = PassiveAggressiveClassifier()

knearestclassifier.fit(x_train, y_train)
decisiontree.fit(x_train, y_train)
logisticregression.fit(x_train, y_train)
passiveAggressive.fit(x_train, y_train)

data1 = {
    "Classification Algorithms": [
        "KNN Classifier",
        "Decision Tree Classifier",
        "Logistic Regression",
        "Passive Aggressive Classifier"
    ],
    "Score": [
        knearestclassifier.score(x_test, y_test),
        decisiontree.score(x_test, y_test),
        logisticregression.score(x_test, y_test),
        passiveAggressive.score(x_test, y_test)
    ]
}

score_df = pd.DataFrame(data1)
print(score_df)
