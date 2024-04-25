import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Загрузим датасет
iris = datasets.load_iris()
X = iris.data

# Найдем оптимальное количество кластеров (сумма квадратов расстояний от каждой точки до ближайшего центроида)
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Построим график метода "локтя"
plt.plot(range(1, 11), wcss)
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()


# Реализуем алгоритм k-means
def kmeans_custom(X, n_clusters, max_iter=300):
    # Инициализация центроидов случайным образом
    centroids = X[np.random.choice(range(len(X)), n_clusters, replace=False)]

    # Находим ближайший к центроиду кластер для каждой точки
    clusters = np.argmin(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)

    # Начинаем итерации
    for _ in range(max_iter):
        # Обновляем центроиды
        new_centroids = np.array([X[clusters == k].mean(axis=0) for k in range(n_clusters)])

        # Проверяем на сходимость
        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

        # Обновляем принадлежность к кластерам
        clusters = np.argmin(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)

        # Визуализация
        plt.scatter(X[:, 0], X[:, 1], c=clusters)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='red')
        plt.title('Шаг {}'.format(_ + 1))
        plt.show()

    return centroids, clusters


# Применяем собственную реализацию
centroids_custom, clusters_custom = kmeans_custom(X[:, :2], n_clusters=3)


