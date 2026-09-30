from smolml.core.ml_array import MLArray
from smolml.core.value import Value
import random

"""
///////////////
/// KMEANS ///
///////////////

Unsupervised clustering algorithm that partitions n samples into k clusters.
Each cluster is represented by the mean of its points (centroid).
Implementation focuses on using MLArray for computations and handling.
"""
class KMeans:
    """
    Implementation of the K-means clustering algorithm using MLArray.
    Partitions data into k clusters by iteratively updating cluster centers
    and reassigning points to the nearest center. Uses Euclidean distance
    for similarity measurement and means for centroid updates.
    """
    def __init__(self, n_clusters, max_iters, tol) -> None:
        """
        Initializes KMeans with the number of clusters, maximum iterations,
        and convergence tolerance. Sets up empty placeholders for centroids
        and cluster assignments.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.centroid_history = []

    def _initialize_centroids(self, X_train):
        """
        Randomly selects k points from the input data to serve as initial
        centroids. Uses random sampling without replacement to ensure
        distinct initial positions.
        """
        centroids = random.sample(X_train.to_list(), self.n_clusters)
        self.centroids = MLArray(centroids)
        self.centroid_history = [self.centroids.to_list()]
        return self.centroids
    
    def _compute_distances(self, X_train):
        """
        Computes the Euclidean distances between all data points and all
        centroids. Uses MLArray broadcasting for efficient computation
        and avoids explicit loops where possible.
        """
        diff = X_train.reshape(-1, 1, X_train.shape[1]) - self.centroids
        squared_diff = diff * diff
        squared_distances = squared_diff.sum(axis=2)
        distances = squared_distances.sqrt()
        return distances
    
    def _assign_clusters(self, distance_matrix):
        """
        Assigns each data point to its nearest centroid based on the
        computed distance matrix. Converts arrays to lists for efficient
        minimum finding and handles the conversion back to MLArray format.
        """
        distances = distance_matrix.to_list()
        labels = []
        
        for sample_distances in distances:
            min_distance = float('inf')
            min_index = 0
            
            for cluster_idx, distance in enumerate(sample_distances):
                if distance < min_distance:
                    min_distance = distance
                    min_index = cluster_idx
                    
            labels.append(min_index)
        
        self.labels_ = MLArray(labels)
        return self.labels_
    
    def _update_centroids(self, X_train):
        """
        Updates centroid positions by computing the mean of all points
        assigned to each cluster. Handles empty clusters by maintaining
        their previous positions. Checks convergence by measuring the
        total movement of all centroids.
        """
        X_data = X_train.to_list()
        labels = self.labels_.to_list()
        new_centroids = []
        
        for cluster_idx in range(self.n_clusters):
            cluster_points = []
            for point_idx, label in enumerate(labels):
                if label == cluster_idx:
                    cluster_points.append(X_data[point_idx])
            
            if cluster_points:
                centroid = []
                n_features = len(cluster_points[0])
                for feature_idx in range(n_features):
                    feature_sum = sum(point[feature_idx] for point in cluster_points)
                    feature_mean = feature_sum / len(cluster_points)
                    centroid.append(feature_mean)
                new_centroids.append(centroid)
            else:
                new_centroids.append(self.centroids.to_list()[cluster_idx])
        
        old_centroids = self.centroids
        self.centroids = MLArray(new_centroids)
        self.centroid_history.append(new_centroids)  # Record new centroids
        
        if old_centroids is not None:
            diff = self.centroids - old_centroids
            movement = (diff * diff).sum().sqrt()
            return movement.data < self.tol
        
        return False

    def fit(self, X_train):
        """
        Main training loop of the KMeans algorithm. Initializes centroids
        and iteratively refines them until convergence or maximum iterations
        are reached. Returns self for method chaining.
        """
        self.centroids = self._initialize_centroids(X_train)
        
        for _ in range(self.max_iters):
            distances = self._compute_distances(X_train)
            self.labels_ = self._assign_clusters(distances)
            has_converged = self._update_centroids(X_train)
            
            if has_converged:
                break
        
        return self
    
    def predict(self, X):
        """
        Predicts cluster assignments for new data points using the trained
        centroids. Raises an error if called before fitting the model.
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet.")
        
        distances = self._compute_distances(X)
        return self._assign_clusters(distances)
    
    def fit_predict(self, X_train):
        """
        Convenience method that performs fitting and prediction in one step.
        Equivalent to calling fit() followed by predict() on the same data.
        """
        return self.fit(X_train).predict(X_train)