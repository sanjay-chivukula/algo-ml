import numpy as np
import matplotlib.pyplot as plt


def k_means_fit(data: np.array, k: int, tolerance: float, max_iterations: int, visualize: bool) -> (dict, dict):
    # Initializes the kmeans model with its parameters
    k_means_obj = KMeans(k, tolerance, max_iterations)

    # Fits the provided date into given number of clusters
    k_means_obj.fit(data, visualize)

    return k_means_obj.centroids, k_means_obj.clusters


class KMeans:
    def __init__(self, k: int, tolerance: float, max_iterations: int) -> None:
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.centroids = None
        self.clusters = None

    def fit(self, data: np.array, visualize: bool) -> None:
        self.centroids = {idx: data[idx] for idx in range(self.k)}

        for i in range(self.max_iterations):
            # Computing feature point distances from the old cluster centroids
            self.clusters = {idx: [] for idx in range(self.k)}

            # main loop
            for feature_point in data:
                distances = [np.linalg.norm(feature_point - centroid) for centroid in self.centroids.values()]
                cluster_idx = distances.index(min(distances))
                self.clusters[cluster_idx].append(feature_point)

            # Temporarily storing old centroid values
            previous_centroids = dict(self.centroids)

            # Computing new cluster centroids
            for cluster_idx in self.clusters.keys():
                self.centroids[cluster_idx] = np.average(self.clusters[cluster_idx], axis=0)

            # Checking for convergence
            for centroid_idx in self.centroids:
                # If something didn't converge continue the main loop
                if not self._is_optimal_cluster(self.centroids[centroid_idx], previous_centroids[centroid_idx]):
                    break
            # If everything converged then break out of the main loop
            else:
                break

        # Show the clusters along with their centroids if the visualization flag is set
        if visualize:
            self._visualize()

    def _is_optimal_cluster(self, current_centroid, previous_centroid) -> bool:
        return np.sum((current_centroid - previous_centroid) / previous_centroid * 100.0) > self.tolerance

    def _visualize(self) -> None:
        colors = ["g", "r", "c", "b", "k"] * 10

        # plotting centroids
        for centroid_pos in self.centroids.values():
            plt.scatter(centroid_pos[0], centroid_pos[1], s=130, marker="X")

        # plotting cluster data points
        for cluster_idx, cluster_points in self.clusters.items():
            color = colors[cluster_idx]
            for feature_point in cluster_points:
                plt.scatter(feature_point[0], feature_point[1], color=color, s=30)

        # displaying the plot
        plt.show()


def test_driver():
    sample_data = np.array([[1, 11],
                            [3, 3],
                            [6, 8],
                            [5, 8],
                            [6, 7],
                            [2, 10],
                            [6, 7],
                            [2, 2]])
    centroids, clusters = k_means_fit(sample_data, 3, 0.0001, 500, True)
    for centroid, cluster in zip(centroids.values(), clusters.values()):
        print(centroid, cluster)


if __name__ == "__main__":
    test_driver()
