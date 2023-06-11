'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Kat Morrocu
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        '''Analysis constructor'''
        self.data = data
        plt.rcParams.update({'font.size': 20})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.'''
        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.'''
        return np.min(self.data.select_data(headers, rows), axis = 0)

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.'''
        return np.max(self.data.select_data(headers, rows), axis = 0)

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.'''
        return (self.min(headers, rows), (self.max(headers, rows)))

    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).
        '''
        return np.sum(self.data.select_data(headers, rows), 0)/len(self.data.select_data(headers, rows))

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.'''
        return np.sum(((self.data.select_data(headers, rows) - self.mean(headers, rows))**2), axis = 0) / (len(self.data.select_data(headers, rows)) - 1)

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.'''
        return np.sqrt(self.var(headers, rows))

    def show(self):
        '''Simple wrapper function for matplotlib's show function.'''
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.'''
        plt.scatter(self.data.select_data([ind_var]), self.data.select_data([dep_var]))
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        return self.data.select_data([ind_var]).flatten(), self.data.select_data([dep_var]).flatten()

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.'''
        num_vars = len(data_vars)
        # Create a new figure with a grid of subplots
        fig, axes = plt.subplots(num_vars, num_vars, figsize=fig_sz, sharex='col', sharey='row')

        # Iterate over the rows and columns of the grid
        for i in range(num_vars):
            for j in range(num_vars):
                # If the row index is equal to the column index, plot a histogram
                if i == j:
                    axes[i, j].hist(self.data.select_data([data_vars[i]]))
                # Otherwise, plot a scatter plot
                else:
                    axes[i, j].scatter(
                        self.data.select_data([data_vars[j]]),
                        self.data.select_data([data_vars[i]])
                    )
                # Set x label for the last row
                if i == num_vars - 1:
                    axes[i, j].set_xlabel(data_vars[j])
                # Set y label for the first column
                if j == 0:
                    axes[i, j].set_ylabel(data_vars[i])

        fig.suptitle(title)
        plt.tight_layout()
        return fig, axes

'''
def main():
    from data import Data
    iris_filename = 'data/iris.csv'
    iris_data = Data(iris_filename)
    an = Analysis(iris_data)
    all_mins, all_maxs = an.range(['sepal_length', 'sepal_width'])
    some_mins, some_maxs = an.range(['sepal_length', 'sepal_width'], rows=np.arange(10))
    print(f"Your range for sepal vars (all samples) is\nmins:{all_mins}\nmaxs:{all_maxs}\nand should be\nmins:[4.3 2. ]\nmaxs:[7.9 4.4]\n")
    print(f"Your range for sepal vars (1st 10 samples) is\nmins:{some_mins}\nmaxs:{some_maxs}\nand should be\nmins:[4.4 2.9]\nmaxs:[5.4 3.9]\n")
    print(f"Your min shape is {all_mins.shape}\nand should be (2,)")
    print(f"Your max shape is {all_maxs.shape}\nand should be (2,)")

if __name__ == "__main__":
    main()
'''
