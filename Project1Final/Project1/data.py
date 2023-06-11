'''data.py
Reads CSV files, stores data, access/filter data by variable name
Kat Morrocu
CS 251 Data Analysis and Visualization
Spring 2023
'''

import csv
import numpy as np

class Data:
    def __init__(self, filepath=None, headers=None, data=None, header2col=None):
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables
            (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
        '''

        self.filepath = filepath
        self.headers = headers
        self.data = data
        self.header2col = header2col

        if filepath != None:
            self.read(filepath)

    def read(self, filepath):
        '''Read the data file (.csv format) located at `filepath` and store it as a 2D numpy ndarray called `self.data`.

        The format of `self.data`:
            - Rows correspond to data samples.
            - Columns correspond to variables/features.

        Parameters:
        -----------
        filepath: str or None. Path to the data .csv file.

        Returns:
        -----------
        None.
        '''

        self.filepath = filepath
        self.headers = []
        self.header2col = {}
        self.data = []

        with open(filepath, 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            line1 = next(csv_reader)
            line2 = next(csv_reader)

            # Check if line2 contains numeric values, raise exception if it does
            for element in line2:
                try:
                    float(element)
                    raise ValueError("Numeric values are not allowed in line2.")
                except ValueError:
                    continue

            line3 = next(csv_reader)
            i = 0
            j = 0
            numlist = []
            
            # Process line3 to extract numeric values and headers
            for element in line3:
                try:
                    numlist.append(float(element))
                    self.headers.append(line1[i])
                    self.header2col[line1[i].strip()] = j
                    j += 1
                except ValueError:
                    pass
                i += 1

            self.data.append(numlist)

            # Process the remaining rows of the CSV file
            for row in csv_reader:
                numlist = []
                for val in row:
                    try: 
                        numlist.append(float(val))
                    except ValueError:
                        continue
                self.data.append(numlist)

        self.data = np.array(self.data)

    def __str__(self):
        '''toString method

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''

        string =  'the first 5 rows of data are: ' + str(self.data[0:5, : ])
        return string

    def get_headers(self):
        '''Get method for headers

        Returns:
        -----------
        Python list of str.
        '''

        return self.headers

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''

        return self.header2col

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''

        return np.size(self.data)

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''

        return len(self.data)

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''

        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''

        indexList = []
        for i in headers:
            indexList.append(self.header2col[i])
        return indexList

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
        '''

        return np.copy(self.data)

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''

        return self.data[0:5, : ]

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''

        return self.data[-5:, : ]

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.
        '''

        self.data = self.data[start_row : end_row, : ]
        return self.data

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.
        '''

        headerIndexes = []

        for i in headers:
            if i in self.header2col:
                headerIndexes.append(self.header2col[i])

        if rows == []:
            return self.data[ : , headerIndexes]
        else:
            return self.data[np.ix_(rows, headerIndexes)]