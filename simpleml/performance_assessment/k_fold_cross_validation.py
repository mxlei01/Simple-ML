"""Implements KFoldCrossValidation."""

from data_extraction.convert_numpy import ConvertNumpy
from performance_assessment.predict_output import PredictOutput
from performance_assessment.residual_sum_squares import ResidualSumSquares


class KFoldCrossValidation:

    """Class for K Fold Cross Validation.

    Class for K Fold Cross Validation for selecting best parameters.

    Attributes:
        convert_numpy (ConvertNumpy): Pandas to Numpy conversion class.
        predict_output (PredictOutput): Output prediction.
        residual_sum_squares (ResidualSumSquares): Computes residual sum of squares.

    """

    def __init__(self):
        """Set up KFoldCrossValidation class with multiple classes.

        Constructor for KFoldCrossValidation, used to setup numpy conversion, output prediction, and residual sum
        of squares.

        """
        self.convert_numpy = ConvertNumpy()
        self.predict_output = PredictOutput()
        self.residual_sum_squares = ResidualSumSquares()

    def k_fold_cross_validation(self, k, model, model_parameters, data_parameters):
        """Perform K Fold Cross Validation.

        Takes in our data, and splits the data to smaller subsets, and these smaller subsets are used as validation
        sets, and everything else not included in the validation set is used as training sets. The model will be
        trained using the training set, and the performance assessment such as RSS would be used on the validation
        set against the model.

        Args:
            k (int): Number of folds.=
            model (obj): Model used for k folds cross validation.
            model_parameters (dict): Model parameters to train the specified model.
            data_parameters (dict): A dictionary of data information:
                {
                    data (pandas.DataFrame): Data used for k folds cross validation,
                    output (str): Output name,
                    features (list of str): A list of feature names.
                }

        Returns:
            float: Average validation error.

        """
        # Sum of the validation error, will divide by k (fold) later
        validation_error_sum = 0

        # Loop through each fold
        for i in range(k):
            # Computes validation, and training set
            validation_set, training_set = self.create_validation_training_set(data_parameters["data"], k, i)

            # Convert our pandas frame to numpy to create validation set
            validation_set_matrix, validation_output = self.convert_numpy.convert_to_numpy(validation_set,
                                                                                           data_parameters["features"],
                                                                                           data_parameters["output"], 1)

            # Create a model with Train Set 1 + Train Set 2
            final_weights = self.create_weights(model, model_parameters, training_set, data_parameters)

            # Predict the output of test features
            predicted_output = self.predict_output.regression(validation_set_matrix,
                                                              final_weights)

            # compute squared error (in other words, rss)
            validation_error_sum += self.residual_sum_squares.residual_sum_squares_regression(validation_output,
                                                                                              predicted_output)

        # Return the validation_error_sum divided by fold
        return validation_error_sum / k

    @staticmethod
    def create_validation_training_set(data, k, iteration):
        """Slice data according to k, iteration, and size of data.

        Computes the validation, and training set according to the k number of folds, and the current iteration.

        Args:
            data (pandas.DataFrame): Data used for k folds cross validation.
            k (int): Number of folds.
            iteration (int): Current K fold validation iteration.

        Returns:
            A tuple that contains training set, and validation set:
                (
                    validation_set (pandas.DataFrame): Validation set.
                    training_set (pandas.DataFrame): Training set.
                )

        """
        length_data = len(data)

        # Compute the start section of the current fold
        start = int((length_data * iteration) / k)

        # Compute the end section of the current fold
        end = int((length_data * (iteration + 1)) / k - 1)

        # Get our validation set from the start to the end+1 (+1 since we need to include the end)
        # <Start : end + 1> Validation Set
        validation_set = data[start:end + 1]

        # The Training set the left and the right parts of the validation set
        # < 0       : Start >   Train Set 1
        # < Start   : End + 1 > Validation Set
        # < End + 1 : n >       Train Set 2
        # Train Set 1 + Train Set 2 = All data excluding validation set
        training_set = data[0:start].append(data[end + 1:length_data])

        return validation_set, training_set

    def create_weights(self, model, model_parameters, training_set, data_parameters):
        """Use model to create weights.

        Use model, model parameters, and training set, create a set of coefficients.

        Args:
            model (obj): Model that can be run.
            model_parameters (dict): A dictionary of model parameters.
            training_set (pandas.DataFrame): Train set used for k folds cross validation.
            data_parameters (dict): A dictionary of data information:
                {
                    data (pandas.DataFrame): Data used for k folds cross validation,
                    output (str): Output name,
                    features (list of str): A list of feature names.
                }

        Returns:
            numpy.array: numpy array of weights created by running model.

        """
        # Convert our pandas frame to numpy to create training set
        training_feature_matrix, training_output = self.convert_numpy.convert_to_numpy(training_set,
                                                                                       data_parameters["features"],
                                                                                       data_parameters["output"], 1)

        # Create a model with Train Set 1 + Train Set 2
        return model(model_parameters=model_parameters, feature_matrix=training_feature_matrix, output=training_output)
