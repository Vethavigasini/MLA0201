import math
from collections import Counter

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def knn(train_X, train_y, test_point, k):
    distances = []
    for x, label in zip(train_X, train_y):
        d = euclidean_distance(x, test_point)
        distances.append((d, label))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    labels = [label for _, label in k_nearest]
    return Counter(labels).most_common(1)[0][0]

X_train = [
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7],
    [8, 6]
]

y_train = ['A', 'A', 'A', 'B', 'B', 'B']

test_sample = [4, 4]
k = 3

prediction = knn(X_train, y_train, test_sample, k)

print("Test sample:", test_sample)
print("K value:", k)
print("Predicted class:", prediction)
