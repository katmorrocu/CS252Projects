'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Kat Morrocu
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`
        '''
        
        super().__init__(orig_dataset)
        self.orig_dataset = orig_dataset
        self.data = data

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.
        '''

        dataArray = self.orig_dataset.select_data(headers)
        header2col = dict()
        for i in range(len(headers)):
            header2col[headers[i]] = i
        self.data = data.Data(self.orig_dataset.filepath, headers=headers, data=dataArray, header2col=header2col)

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.
        '''

        projectedData = self.data.get_all_data()
        onesColumn = np.ones([projectedData.shape[0], 1])
        homogeneousData = np.hstack([projectedData, onesColumn])
        return homogeneousData

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.
        '''
        M = len(magnitudes)
        identityMatrix = np.eye(M, M)
        magnitudes = np.expand_dims(np.array(magnitudes), axis=1)
        temporaryMatrix = np.hstack((identityMatrix, magnitudes))
        rowOfZeros = [0] * M + [1]
        return np.vstack((temporaryMatrix, rowOfZeros))

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.
        '''
        M = len(magnitudes)
        scale = np.eye(M + 1, M + 1)
        for i in range(M):
            scale[i,i] = magnitudes[i]
        return scale

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!
        '''

        projectedData = self.get_data_homogeneous()
        translationMatrix = self.translation_matrix(magnitudes)
        translation = (projectedData @ translationMatrix.T)
        modifiedData = np.delete(translation, -1, 1)
        self.data = data.Data(headers = self.data.get_headers(), data = modifiedData, header2col = self.data.get_mappings())
        return self.data

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!
        '''

        projectedData = self.get_data_homogeneous()
        scaleMatrix = self.scale_matrix(magnitudes)
        scaling = (projectedData @ scaleMatrix.T)
        modifiedData = np.delete(scaling, -1, 1)
        self.data = data.Data(headers=self.data.get_headers(), data = modifiedData, header2col = self.data.get_mappings())
        return self.data

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`
        '''

        projectedData = self.get_data_homogeneous()
        transformedData = (projectedData @ C.T)
        modifiedData = np.delete(transformedData, -1, 1)
        self.data = data.Data(headers=self.data.get_headers(), data = modifiedData, header2col=self.data.get_mappings())
        return modifiedData

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        mins = np.amin(self.min(self.data.get_headers()), 0)
        maxs = np.amax(self.max(self.data.get_headers()), 0)
        translationMagnitudes = [-1 * mins] * (self.data.data.shape[1])
        translationMatrix = self.translation_matrix(translationMagnitudes)
        scaleMagnitudes = [1/(maxs-mins)] * (self.data.data.shape[1])
        scaleMatrix = self.scale_matrix(scaleMagnitudes)
        C = scaleMatrix @ translationMatrix
        return self.transform(C)

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        mins = self.min(self.data.get_headers())
        maxs = self.max(self.data.get_headers())
        translationMatrix = self.translation_matrix(-1 * mins)
        scaleMatrix = self.scale_matrix(1/(maxs-mins))
        C = scaleMatrix @ translationMatrix
        transformed = self.transform(C)
        self.data = data.Data(headers=self.data.get_headers(), data = transformed, header2col = self.data.get_mappings())
        return transformed

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.
        '''
        
        headerIndex = self.data.get_header_indices([header])
        identityMatrix = np.eye(self.data.data.shape[1]+1)
        if headerIndex[0] == 0:
            identityMatrix[1:3,1:3] = [[np.cos(np.radians(degrees)), -np.sin(np.radians(degrees))], [np.sin(np.radians(degrees)), np.cos(np.radians(degrees))]]
        elif headerIndex[0] == 1:
            identityMatrix[0,0] = np.cos(np.radians(degrees))
            identityMatrix[0,2] = np.sin(np.radians(degrees))
            identityMatrix[2,0] = -np.sin(np.radians(degrees))
            identityMatrix[2,2] = np.cos(np.radians(degrees))
        else: 
            identityMatrix[0:2,0:2] = [[np.cos(np.radians(degrees)),-np.sin(np.radians(degrees))], [np.sin(np.radians(degrees)), np.cos(np.radians(degrees))]]
        return identityMatrix

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
        '''
        
        projectedData = self.get_data_homogeneous()()
        rotationMatrix = self.create_rotation_matrix_3d(header, degrees)
        rotatedData = np.dot(projectedData, rotationMatrix.T)
        modifiedData = np.delete(rotatedData, -1, axis=1)
        self.data = data.Data(headers=self.data.get_headers(), data=modifiedData, header2col=self.data.get_mappings())
        return data

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
        '''
        
        plt.figure(figsize = (5, 5))
        plt.scatter(self.data.select_data([ind_var]), self.data.select_data([dep_var]), c=self.data.select_data([c_var]), cmap=palettable.colorbrewer.sequential.Blues_7.mpl_colormap)
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.colorbar(label = c_var)