'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Kat Morrocu
CS 251/2 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset. Should be set as an instance variable.
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`
        '''
        dataMeans = np.mean(data, axis = 0)
        centeredData = data - dataMeans
        covarianceMatrix = (1 / (data.shape[0] - 1)) * (centeredData.T @ centeredData)
        return covarianceMatrix

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        totalVariance = np.sum(e_vals)
        proportionVariance = e_vals / totalVariance
        return proportionVariance.tolist()

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        cumulativeVariance = []
        for i in range(len(prop_var)):
            cumulativeVariance.append(np.sum(prop_var[:i + 1]))
        return cumulativeVariance

    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.
        '''
        self.vars = vars
        self.normalized = normalize
        relevant_d = self.data[vars]
        self.A = relevant_d.to_numpy()
        maxVal = np.max(self.A, axis = 0)
        self.max = maxVal
        minVal = np.min(self.A, axis = 0)
        self.min = minVal
        rangeVal = self.max - self.min
        self.range = rangeVal
        meanVal = np.mean(self.A, axis = 0)
        self.mean = meanVal
        covarianceMatrix = self.covariance_matrix(self.A)

        if normalize == True:
            normalizedData = (self.A - self.min)/(self.range)
            self.A = normalizedData
            covarianceMatrix = self.covariance_matrix(normalizedData)
            print('A min: ', minVal) 
            print('A max: ', maxVal)
        
        self.e_vals, self.e_vecs = np.linalg.eig(covarianceMatrix)
        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).
        '''
        if num_pcs_to_keep is None:
            plt.plot(np.arange(len(self.prop_var)), self.cum_var, 'ro-', markersize = 11)
        else:
            plt.plot((np.arange(num_pcs_to_keep)), self.cum_var[:num_pcs_to_keep], 'ro-', markersize = 11)
        
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance')
        plt.title('Variance by Principal Components')

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.
        '''
        return self.A @ self.e_vecs[:, pcs_to_keep]

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)
        '''
        projectedPCA = self.pca_project(np.arange(top_k))

        if self.normalized == False:
            projection = (projectedPCA @ self.e_vecs[:, 0:top_k].T)+ self.mean
        else:
            projection = self.range * (projectedPCA @ self.e_vecs[:, 0:top_k].T) - self.min + self.mean

        return projection