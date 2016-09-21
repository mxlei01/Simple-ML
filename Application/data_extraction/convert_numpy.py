import copy


class ConvertNumpy:
    """For converting Pandas data to Numpy array or matrix.

    The ConvertNumpy class contains useful functions to convert widely known data format, such as Pandas to
    numpy data types.

    """

    @staticmethod
    def convert_to_numpy(pandas_frame, features, output, constant=None):
        """Converts pandas frame to numpy matrix and array.

        Converts a pandas frame to one numpy matrix, and one numpy array. The numpy matrix are the features, and the
        array will be the feature's corresponding output. This will first add a constant if there's
        any to the pandas frame. Then extract the features that we want from the pandas_frame using the features
        array, and get an output from the output array.

        Args:
            pandas_frame (pandas.DataFrame): A pandas frame that we want to convert to numpy.
            features (list): An array of string that indicates the column that we want to extract.
            output (array of str): An array of string that indicates the column that we want to extract for output.
            constant (int): A constant that we want to add.

        Returns:
            A tuple that contains a numpy matrix, and a numpy array:
                (
                    features_numpy (numpy.matrix): A numpy matrix of features extracted from pandas_frame.
                    output_numpy (numpy.array): A numpy array of output extracted from pandas_frame.
                )

        """
        # Make a copy of features, output, and pandas frame
        features = copy.deepcopy(features)
        output = copy.deepcopy(output)
        pandas_frame = pandas_frame.copy(deep=True)

        # If a constant is specified, update the features list with the constant, so that our feature would
        # also extract the constant
        if constant:
            pandas_frame['constant'] = 1
            features = ['constant'] + features

        # Convert the features columns into a matrix
        features_numpy = pandas_frame.as_matrix(columns=features)

        # Convert the output columns into a matrix, and transpose it so that it becomes a list instead lists of lists,
        # the index [0] takes the only item from the double list into a list
        output_numpy = pandas_frame.as_matrix(columns=output).transpose()[0]

        return features_numpy, output_numpy
