'''kmeans.py
Performs K-Means clustering
YOUR NAME HERE
CS 251: Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return np.copy(self.data)

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.
        '''
        return np.linalg.norm(pt_1 - pt_2)

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.linalg.norm(pt - centroids, axis = 1)

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        index = np.random.random_integers(0 , len(self.data) - 1, k)
        self.centroids = self.data[index]
        return self.centroids 

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for
        '''
        self.initialize(k)

        for i in self.centroids:
            self.data_centroid_labels = self.update_labels(self.centroids)
            self.centroids, diff = self.update_centroids(k, self.data_centroid_labels, self.centroids)
            if np.all(np.abs(diff) < tol):
                break
        
        self.inertia = self.compute_inertia()
        return self.inertia

    def cluster_batch(self, k=2, n_iter=1, verbose=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        lowestInertia  = 1000
        for i in range(n_iter):
            inertia = self.cluster(k)
            if inertia < lowestInertia:
                centroids = self.centroids
                centroid_labels = self.data_centroid_labels
                lowestInertia = inertia
        self.centroids = centroids
        self.data_centroid_labels = centroid_labels
        self.inertia = lowestInertia

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        cluster = []

        for i in range(len(self.data)):
            euclediantDistance = self.dist_pt_to_centroids(self.data[i], centroids = centroids)
            assignCluster = np.argmin(euclediantDistance, axis = 0)
            cluster.append(assignCluster)

        return np.asarray(cluster)

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values
        '''
        self.k = k
        newCentroids = np.zeros([prev_centroids.shape[0], prev_centroids.shape[1]])

        for i in range(k):
            index = np.where(data_centroid_labels == i)
            myArray = np.array(index)
            if myArray.size != 0:
                centroid = np.sum(self.data[index,:],axis = 1) / int(myArray.size)
                newCentroids[i, :] = centroid 
            else:
                continue

        self.centroids=newCentroids
        centroid_diff = newCentroids-prev_centroids
        return newCentroids, centroid_diff    

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        sumDistance = 0

        for i in range(self.k):
            indices = np.where(self.data_centroid_labels == i)[0]  # Use [0] to access the indices array
            for j in indices:
                sumDistance += (self.dist_pt_to_pt(self.data[j, :], self.centroids[i, :])) ** 2

        inertia = sumDistance / self.data.shape[0]
        self.inertia = inertia
        return inertia

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.'''
        colors = ['red', 'green', 'blue', 'orange', 'purple']

        for i in range(self.k):
            data = self.data[self.data_centroid_labels == i]
            #centroids = self.data[self.centroids]
            plt.scatter(data[:,0], data[:,1], c = [colors[i]])
            plt.scatter(self.centroids[:,0], self.centroids[:,1] , c = 'k', marker= '*', label = 'Centroids')

    def elbow_plot(self, max_k, n_iter = 1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        '''
        inertias = []
        for k in range(1, max_k+1):
            total_inertia = 0
            for _ in range(n_iter):
                inertia = self.cluster_batch(k=k)
                total_inertia += self.inertia
            average_inertia = total_inertia / n_iter
            inertias.append(average_inertia)
        
        plt.plot(range(1, max_k+1), inertias, 'ro-')
        plt.xlabel('K Clusters')
        plt.ylabel('Inertia')
        plt.title('K Clusters vs Inertia')
        plt.show()

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        for i in range(self.k):
            index = np.where(self.data_centroid_labels == i)
            self.data[index] = self.centroids[i]
