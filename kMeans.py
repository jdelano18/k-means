import random
import numpy as np
from utils import euclidean_distance

class kMeans():

    def __init__(self, k, n_iter=100):
        self.k = k
        self.n_iter = n_iter

    def _initialize_centroids(self, X):
        """
        Returns a matrix representing k randomly chosen instances for the initial centroids
        """
        n_instances, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))

        # use random.sample to avoid picking the same instance
        random.seed(4)
        random_indices = random.sample(range(n_instances), self.k)

        for i, instance in enumerate(random_indices):
            centroids[i] = X[instance]

        return centroids

    def _find_nearest_centroid(self, instance, centroids):
        """
        Helper method for create_clusters.

        Returns the index of the closest centroid for a given instance
        Distance measured using euclidean_distance
        """
        closest = -1
        closest_dist = float('inf')
        for i, c in enumerate(centroids):
            dist = euclidean_distance(instance, c)
            if dist < closest_dist:
                closest_dist = dist
                closest = i
        return closest

    def _create_clusters(self, X, centroids):
        """
        Returns a list of k-lists, each containing the indices of instances that are closest to the centroid

        ** stop storing the instances themselves it wastes space, just save indices
        """
        n_instances, n_features = np.shape(X)
        clusters = [[] for _ in range(self.k)]  # create clusters as empty lists and append indices to it

        for i, x_i in enumerate(X):
            centroid_idx = self._find_nearest_centroid(x_i, centroids)
            clusters[centroid_idx].append(i)

        assert sum([len(c) for c in clusters]) == n_instances # sanity check

        return clusters

    def _update_centroids(self, X, clusters):
        """
        Calculates a new centroid/centers by computing the mean of each cluster (by attribute value)
        Returns matrix of k updated centroids that are each len(n_features)
        """

        n_features = np.shape(X)[1]
        new_centroids = np.zeros((self.k, n_features))

        for i, clstr in enumerate(clusters):
            centroid = np.mean(X[clstr], axis=0) # select all rows with indices in cluster, axis=0 --> cols
            new_centroids[i] = centroid
        return new_centroids

    def train(self, X):
        centroids = self._initialize_centroids(X) # pick k random centroids to start

        for _ in range(self.n_iter):
            clusters = self._create_clusters(X, centroids) # generate clusters from centroids

            prev_centroids = centroids # hold onto previous

            centroids = self._update_centroids(X, clusters) # update new centroids with current

            # If no centroids have changed => convergence
            delta = centroids - prev_centroids
            if np.all((delta == 0)):
                break

        self.final_clusters = clusters


    def _generate_prediction_groups(self, X):
        """
        Return a vector of len n_instances that correspond to 0, 1, ... k
        that corresponds to the cluster each instance was in at the end of training
        """
        preds = np.zeros(np.shape(X)[0])
        for i, c in enumerate(self.final_clusters):
            for instance_index in c:
                preds[instance_index] = i
        return preds

    def _generate_predition_map(self, X, y):
        """
        *Note:* uses true y value -- keep in seperate function to avoid issues
        This function returns a dict from our cluster numbers 0, 1, ... k -> actual class values
        """
        result = {n : 0 for n in range(self.k)}         # map clusters to classes
        for i, c in enumerate(self.final_clusters):

            # count number of each class per cluster to find most popular
            class_map = {class_val : 0 for class_val in y}  # maps class_val -> count
            for instance in c:
                val = y[instance] # look up actual val from y
                class_map[val] = class_map.get(val, 0) + 1 # update counts

            most_popular_class = max(class_map, key=class_map.get)
            result[i] = most_popular_class
        return result

    def predict(self, X, y):
        preds = self._generate_prediction_groups(X)
        pred_map = self._generate_predition_map(X, y)
        results = [pred_map[p] for p in preds] # apply map to change cluster-index 0...k to actual class_vals
        return results
