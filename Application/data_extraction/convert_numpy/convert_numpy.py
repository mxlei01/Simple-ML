import copy

class ConvertNumpy():
    # Usage:
    #   A class for converting Pandas data to Numpy array or matrix

    def convert_to_numpy(self, pandas_frame, features, output, constant=None):
        # Usage:
        #       Converts a pandas frame to one numpy matrix, and one numpy array. This will
        #       first add a constant if there's any to the pandas frame. Then extract the features
        #       that we want from the pandas_frame using the features array, and get an output
        #       from the output array.
        # Arguments:
        #       pandas_frame (pandas frame)   : a pandas frame that we want to convert to numpy
        #       features     (array)          : an array of string that indicates the column
        #                                       that we want to extract
        #       output       (array)          : an array of string that indicates the column
        #                                       that we want to extract for output
        #       constant     (int)            : a constant that we want to add
        # Return:
        #       features_numpy (numpy matrix) : a numpy matrix of features extracted from pandas_frame
        #       output_numpy   (numpy array)  : a numpy array of output extracted from pandas_frame

        # Make a copy of features, output, and pandas frame
        features = copy.deepcopy(features)
        output = copy.deepcopy(output)
        pandas_frame = pandas_frame.copy(deep=True)

        # If constant is not none
        if constant:

            # Add a column called constant, and the values are all 1
            pandas_frame['constant'] = 1

            # Add the constant to the features array
            features = ['constant'] + features

        # Build a features numpy matrix
        features_numpy = pandas_frame.as_matrix(columns=features)

        # Build a output numpy array
        output_numpy = pandas_frame.as_matrix(columns=output).transpose()[0]

        # Return as a tuple
        return features_numpy, output_numpy
