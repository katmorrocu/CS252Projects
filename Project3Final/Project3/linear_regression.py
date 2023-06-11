'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Kat Morrocu
CS251 Data Analysis Visualization
Spring 2023
'''

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean SEE. float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        ind_vars = self.data.select_data(ind_vars)
        dep_var = self.data.select_data([dep_var])
        self.y = dep_var

        # Create the design matrix by concatenating the independent variables with a column of ones
        self.A = np.hstack([ind_vars, np.ones([ind_vars.shape[0], 1])])

        # Perform least squares regression and get coefficients
        c, _, _, _ = scipy.linalg.lstsq(self.A, self.y)

        # Remove the last column from the design matrix (column of ones)
        self.A = np.delete(self.A, -1, 1)

        # Store the coefficients (slopes) and intercept
        self.slope = c[:-1]
        self.intercept = c[-1, -1]

        # Predict the dependent variable values
        y_pred = self.predict(self.A)

        # Calculate the R^2 statistic
        self.R2 = self.r_squared(y_pred)

        # Compute the residuals
        self.residuals = self.compute_residuals(y_pred)

    def predict(self, X = None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        '''
        if X is None:
            X = self.A
        if self.p > 1:
            X = self.make_polynomial_matrix(X, self.p)
        y_pred = (X @ self.slope) + self.intercept
        return y_pred

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        mean_Y = np.mean(self.y)
        smd = np.sum((self.y - mean_Y)**2)
        res = np.sum((self.y - y_pred)**2)
        R2 = 1 - (res/smd)
        return R2

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        residuals = self.y - y_pred
        return residuals

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error
        '''
        residuals = self.compute_residuals(self.predict())
        mse = np.mean(residuals**2)
        return mse

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot
        '''

        (x, y) = super().scatter(ind_var, dep_var, title)
        xdata = x.reshape(x.shape[0], 1)
        x_range = np.linspace(xdata[:, 0].min(), xdata[:, 0].max())

        if self.p > 1:
            # Add an extra dimension for polynomial matrix
            x_range = x_range[:, np.newaxis]

            # Create polynomial matrix and calculate polynomial line
            Ap = self.make_polynomial_matrix(x_range, self.p)
            polyLine = np.squeeze(self.intercept + Ap @ self.slope)

            # Plot polynomial line
            plt.plot(x_range, polyLine, 'g')

        else:
            # Calculate and plot linear regression line
            yLine = self.intercept + self.slope * x_range
            plt.plot(x_range, yLine.reshape(yLine.shape[1], yLine.shape[0]), 'r')

        plt.title(title + ' | R2 value: ' + str(self.R2))

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.
        '''
        fig, axes = super().pair_plot(data_vars, fig_sz, hists_on_diag)

        # Loop over rows and columns of the pair plot grid
        for row in range(len(data_vars)):
            for col in range(len(data_vars)):

                # Perform linear regression for the current variable pair
                self.linear_regression([data_vars[col]], data_vars[row])

                # Generate x values for the regression line
                x = np.linspace(self.data.select_data([data_vars[col]])[:, 0].min(), self.data.select_data([data_vars[col]])[:, 0].max())

                # Calculate y values for the regression line
                y = self.intercept + self.slope * x

                axes[row, col].plot(x, y.reshape(y.shape[1], y.shape[0]), 'r')
                axes[row, col].set_title('R^2= {:.3f}'.format(self.R2))

                # If it's a diagonal subplot and histograms are enabled, plot the histogram
                if row == col and hists_on_diag:

                    # Get the total number of variables
                    varNum = len(data_vars)

                    # Remove the current subplot
                    axes[row, col].remove()

                    # Create a new subplot at the same position
                    axes[row, col] = fig.add_subplot(varNum, varNum, row * varNum + col + 1)

                    # Plot the histogram of the current variable
                    axes[row, col].hist(self.data.select_data([data_vars[row]]))

                    # Set x-axis ticks and label
                    if col < varNum - 1:
                        axes[row, col].set_xticks([])
                    else:
                        axes[row, col].set_xlabel(data_vars[row])

                    # Set y-axis ticks and label
                    if row > 0:
                        axes[row, col].set_yticks([])
                    else:
                        axes[row, col].set_ylabel(data_vars[row])

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.
        '''
        matrix = A
        for powr in range(2,p + 1):
            newP = np.power(A, powr)
            matrix = np.hstack((matrix , newP))
        return matrix

    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).
        '''
        self.ind_vars = ind_var
        self.dep_var = dep_var
        ind_vars = self.data.select_data(ind_var)
        dep_var = self.data.select_data([dep_var])
        self.y = dep_var
        self.p = p
        self.A = ind_vars

        # Create the polynomial matrix
        Mp = self.make_polynomial_matrix(ind_vars, p)

        # Add a column of ones for homogeneous coordinates
        A1 = np.hstack([Mp, np.ones([self.A.shape[0], 1])])

        # Perform least squares regression
        c, _, _, _ = scipy.linalg.lstsq(A1, self.y)

        # Store the coefficients (slopes) and intercept
        self.slope = c[:-1]
        self.intercept = c[-1, -1]

        # Predict the dependent variable values
        y_pred = self.predict(self.A)

        # Calculate the R^2 statistic
        self.R2 = self.r_squared(y_pred)

        # Compute the residuals
        self.residuals = self.compute_residuals(y_pred)

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.intercept = intercept
        self.slope = slope
        self.p = p
        self.y = self.data.select_data([dep_var])
        self.A = self.data.select_data(ind_vars)
        y_pred = self.predict()
        self.R2 = self.r_squared(y_pred)
        self.residuals = self.compute_residuals(y_pred)
