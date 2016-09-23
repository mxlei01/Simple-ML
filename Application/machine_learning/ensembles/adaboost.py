import numpy as np
import math
import pandas as pd


class AdaBoost:
    """Class that implements Adaboost for decision tree and logistic regression.

    AdaBoost is based on setting a weight α on each training sample, then for each iteration:
        1. Generate another model f_t(x) with data weights α.
        2. Compute coefficient w_t, where w_t=1/2*ln((1-weighted_error(f_t)/(weighted_error(f_t)).
                                    where weighted_error(f_t)=   Total weight of mistakes
                                                              -------------------------------
                                                              Total weight of all data points
        3. Recompute weights α_i, where α_i = α_i*e^-w_t, if f_t(x)=y_i (the prediction is correct).
                                              α_i*e^w_t, if f_t(x)=/=y_i (the prediction is not correct).
        4. Normalize the weights α_i =        α_i
                                      ---------------------
                                      sum(α_j, for all α's)

    After T iteration (we can set this do a max number), we can predict the final result by:
        y=sign(Σ, T, n=1, w_t*f_t(x)).

    Essentially for an input, we use f_t(x) to determine an output (-1, or +1 for binary decision trees), and
    we multiply the weight for that f_t(x). Each f_t(x) has it's own weight. Then using the sign, if the result
    is positive, then our output is +1, else -1.

    """

    @staticmethod
    def decision_tree(data, features, target, iterations, predict_method, model, model_method, model_parameters):
        """Decision Tree ensemble learning algorithm for Adaboost.

        Uses a decision tree model, applies the adaboost algorithm and generates T number of models through the
        iterations parameter.

        Args:
            data (pandas.DataFrame): Training data.
            features (list of str): List of features that we want to train.
            target (str): The target (output) that we want to train.
            iterations (int): Number of iterations.
            predict_method  (func): Function to predict output.
            model (obj): A model that contains a predict function to predict the output based on some input features.
            model_method (function): Model's function to generate a model.
            model_parameters (dict): Model parameters for the model, such as depth to train for a decision tree.

        Returns:
            A list of tuples (weights, model):
                [
                    (
                        list of float: List of weights for the coefficients.
                        obj: Model trained.
                    )
                ]

        """
        # Each row of data (training data), has an alpha
        alpha = pd.Series(data=[1]*len(data))

        # Initialize a list of weights
        weights_list = []

        # Initialize a list of models
        models_list = []

        # Get the column of data that includes our training output
        target_values = data[target]

        # Loop through each iteration, and generate one model per generation
        for _ in range(iterations):
            # Use the model to generate a model, the output will be a decision tree
            generated_model = getattr(model, model_method)(**{**{"data": data, "features": features, "target": target,
                                                                 "data_weights": alpha},
                                                              **model_parameters})

            # Insert the new model to the models list
            models_list.append(generated_model)

            # Make predictions
            predictions = data.apply(lambda x, gm=generated_model: predict_method(gm, x), axis=1)

            # Creating an array of boolean values indicating if each data was correctly classified
            correct = predictions == target_values
            wrong = predictions != target_values

            # Compute the weighted_error(f_t(x))
            # Weighted Error =    total weight of mistakes
            #                  -------------------------------
            #                  total weight of all data points
            # Best Possible Error: 0, Worst: 1.0, Random Classifier: 0.5
            weighted_error = sum(alpha[wrong])/sum(alpha)

            # Compute w_t = w_t=1*ln(1-weighted_error(f_t))
            #                   -    ---------------------
            #                   2     weighted_error(f_t)
            # If f_t(x) classifier was a good classifier, then the weighted error will be low, which
            # results in a higher w_t for that classifier. If the classifier was a bad classifier, then the result
            # would be a classifier with lower w_t
            weight = 0.5 * math.log((1-weighted_error)/weighted_error)

            # Add the new weight to our weights list
            weights_list.append(weight)

            # Recompute weights α_i, where α_i = α_i*e^-w_t, if f_t(x)=y_i (the prediction is correct)
            #                                    α_i*e^w_t, if f_t(x)=/=y_i (the prediction is not correct)
            # If f_t classifier got the prediction correct at a point:
            #   If w_t is high (2.3), then we multiply α by e^(-2.3)=0.1, which means we will decrease
            #   the importance of α for that specific row of data.
            #   If w_t is low (0), then we will multiply α by e^(0)=1, then we will keep the importance is the same.
            # If f_t classifier got the prediction incorrect at a point:
            #   If w_t is high (2.3), then we multiply α by e^(2.3)=9.98, which means we will increase the importance
            #   of α for that specific row of data.
            #   If w_t is low (0), then we will multiply α by e^(0)=1, then we will keep the importance the same.
            exponential_weight = correct.apply(lambda is_correct, w=weight: math.exp(-w) if is_correct else math.exp(w))

            # Scale alpha by multiplying by exponential_weight
            alpha = alpha*exponential_weight

            # Normalize the weights α_i =        α_i
            #                            ---------------------
            #                            sum(α_j, for all α's)
            alpha = alpha/sum(alpha)

        return weights_list, models_list

    @staticmethod
    def logistic_regression(feature_matrix, label, iterations, predict_method, model, model_method, model_parameters):
        """Logistic regression ensemble learning algorithm for Adaboost.

        Uses a logistic regression model, applies the adaboost algorithm and generates T number of models through the
        iterations parameter.

        Args:
            feature_matrix (numpy.matrix): Features of a dataset.
            label (numpy.array): The label of a dataset.
            iterations (int): Number of iterations.
            predict_method (func): Function to predict output.
            model (obj): A model that contains a predict function to predict the output based on some input features.
            model_method (function): Model's function to generate a model.
            model_parameters (dict): Model parameters for the model, such as step size.

        Returns:
            A list of tuples (weights, model):
                [
                    (
                        list of float: List of weights for the coefficients.
                        obj: Model trained.
                    )
                ]

        """
        # Each row of data (training data), has an alpha
        alpha = np.array([1]*len(feature_matrix))

        # Initialize a list of weights
        weights_list = []

        # Initialize a list of models
        models_list = []

        # Loop through each iteration, and generate one model per generation
        for _ in range(iterations):
            # Use the model to generate a model, the output will be coefficients
            generated_model = getattr(model, model_method)(**{**{"feature_matrix": feature_matrix, "label": label,
                                                                 "weights_list": alpha},
                                                              **model_parameters})

            # Insert the new model to the models list
            models_list.append(generated_model)

            # Make predictions
            predictions = predict_method(feature_matrix, generated_model)

            # Creating an array of boolean values indicating if each data was correctly classified
            correct = predictions == label
            wrong = predictions != label

            # Compute the weighted_error(f_t(x))
            # Weighted Error =    total weight of mistakes
            #                  -------------------------------
            #                  total weight of all data points
            # Best Possible Error: 0, Worst: 1.0, Random Classifier: 0.5
            weighted_error = sum(alpha[wrong]) / sum(alpha)

            # Compute w_t = w_t=1*ln(1-weighted_error(f_t))
            #                   -    ----------------------
            #                   2     weighted_error(f_t)
            # If f_t(x) classifier was a good classifier, then the weighted error will be low, which
            # results in a higher w_t for that classifier. If the classifier was a bad classifier, then the result
            # would be a classifier with lower w_t
            weight = 0.5 * math.log((1 - weighted_error) / weighted_error)

            # Add the new weight to our weights list
            weights_list.append(weight)

            # Recompute weights α_i, where α_i = α_i*e^-w_t, if f_t(x)=y_i (the prediction is correct)
            #                                    α_i*e^w_t, if f_t(x)=/=y_i (the prediction is not correct)
            # If f_t classifier got the prediction correct at a point:
            #   If w_t is high (2.3), then we multiply α by e^(-2.3)=0.1, which means we will decrease
            #   the importance of α for that specific row of data.
            #   If w_t is low (0), then we will multiply α by e^(0)=1, then we will keep the importance is the same.
            # If f_t classifier got the prediction incorrect at a point:
            #   If w_t is high (2.3), then we multiply α by e^(2.3)=9.98, which means we will increase the importance
            #   of α for that specific row of data.
            #   If w_t is low (0), then we will multiply α by e^(0)=1, then we will keep the importance the same.
            exponential_weight = np.array([math.exp(-weight) if i else math.exp(weight) for i in correct])

            # Scale alpha by multiplying by exponential_weight
            alpha = alpha * exponential_weight

            # Normalize the weights α_i =        α_i
            #                            ---------------------
            #                            sum(α_j, for all α's)
            alpha = alpha / sum(alpha)

        return weights_list, models_list
