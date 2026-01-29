import math
import random

def gaussian_pdf(x, mean, var):
    if var <= 1e-12:
        var = 1e-12
    coeff = 1.0 / math.sqrt(2.0 * math.pi * var)
    expo = math.exp(-((x - mean) ** 2) / (2.0 * var))
    return coeff * expo

def em_gmm_1d(data, k=2, max_iter=50, tol=1e-6):
    n = len(data)

    means = random.sample(data, k)
    vars_ = [1.0 for _ in range(k)]
    pis = [1.0 / k for _ in range(k)]

    prev_ll = None

    for it in range(max_iter):
        resp = [[0.0] * k for _ in range(n)]

        for i, x in enumerate(data):
            denom = 0.0
            for j in range(k):
                resp[i][j] = pis[j] * gaussian_pdf(x, means[j], vars_[j])
                denom += resp[i][j]
            if denom == 0.0:
                for j in range(k):
                    resp[i][j] = 1.0 / k
            else:
                for j in range(k):
                    resp[i][j] /= denom

        for j in range(k):
            Nj = sum(resp[i][j] for i in range(n))
            if Nj == 0.0:
                Nj = 1e-12

            new_mean = sum(resp[i][j] * data[i] for i in range(n)) / Nj
            new_var = sum(resp[i][j] * ((data[i] - new_mean) ** 2) for i in range(n)) / Nj
            new_pi = Nj / n

            means[j] = new_mean
            vars_[j] = new_var if new_var > 1e-12 else 1e-12
            pis[j] = new_pi

        ll = 0.0
        for x in data:
            s = 0.0
            for j in range(k):
                s += pis[j] * gaussian_pdf(x, means[j], vars_[j])
            ll += math.log(s + 1e-12)

        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    labels = []
    for i in range(n):
        best = max(range(k), key=lambda j: resp[i][j])
        labels.append(best)

    return means, vars_, pis, labels

if __name__ == "__main__":
    data = [
        1.0, 1.2, 0.8, 1.1, 0.9, 1.3,
        5.0, 5.1, 4.9, 5.2, 4.8, 5.3
    ]

    means, vars_, pis, labels = em_gmm_1d(data, k=2, max_iter=100)

    print("Final Parameters:")
    for j in range(2):
        print("Cluster", j)
        print("  Mean =", round(means[j], 4))
        print("  Var  =", round(vars_[j], 4))
        print("  Pi   =", round(pis[j], 4))

    print("\nData Point -> Cluster")
    for x, c in zip(data, labels):
        print(x, "->", c)
