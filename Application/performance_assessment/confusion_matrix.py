import pandas as pd


class ConfusionMatrix:
    # Usage:
    #   Used for computing a confusion matrix

    def confusion_matrix(self, output, predicted_output):
        # Usage:
        #       Produces a confusion matrix, where:
        #           True Label: +1, Predicted Label: +1 = True Positive
        #           True Label: +1, Predicted Label: -1 = False Negative
        #           True Label: -1, Predicted Label: +1 = False Positive
        #           True Label: -1, Predicted Label: -1 = True Negative
        # Arguments:
        #       output           (numpy array)   : a numpy array containing real output
        #       predicted_output (numpy array)   : a numpy array containing predicted output
        # Return:
        #       confusion matrix (dict)          : a dictionary containing keywords true_positives, false_negatives,
        #                                          false_positives, true_negatives

        # Convert the output and predicted_output to pandas matrix
        df = pd.concat([pd.Series(output, name="output"), pd.Series(predicted_output, name="predicted_output")], axis=1)

        # Count True Positives
        true_positives = sum(df.apply(lambda x: x["output"] == 1 and x["predicted_output"] == 1, axis=1))

        # Count False Negatives
        false_negatives = sum(df.apply(lambda x: x["output"] == 1 and x["predicted_output"] == -1, axis=1))

        # Count False Positives
        false_positives = sum(df.apply(lambda x: x["output"] == -1 and x["predicted_output"] == 1, axis=1))

        # Count True Negatives
        true_negatives = sum(df.apply(lambda x: x["output"] == -1 and x["predicted_output"] == -1, axis=1))

        # Return the dictionary
        return {"true_positives": true_positives, "false_negatives": false_negatives,
                "false_positives": false_positives, "true_negatives": true_negatives}

    def precision(self, output, predicted_output):
        # Usage:
        #       Computes the precision of the output and predicted_output, where
        #               precision =         # true positives
        #                           ------------------------------------
        #                           # true positives + # false negatives
        # Arguments:
        #       output           (numpy array)   : a numpy array containing real output
        #       predicted_output (numpy array)   : a numpy array containing predicted output
        # Return:
        #       precision        (float) : precision percentage

        # Get the confusion matrix
        confusion_matrix = self.confusion_matrix(output, predicted_output)

        # Compute and return precision
        return float(confusion_matrix["true_positives"])/float(confusion_matrix["true_positives"] +
                                                               confusion_matrix["false_negatives"])

    def recall(self, output, predicted_output):
        # Usage:
        #       Computes the recall of the output and predcited_output, where
        #               recall =        # true positives
        #                       ------------------------------------
        #                       # true positives + # false positives
        # Arguments:
        #       output           (numpy array)   : a numpy array containing real output
        #       predicted_output (numpy array)   : a numpy array containing predicted output
        # Return:
        #       recall           (float) : precision percentage

        # Get the confusion matrix
        confusion_matrix = self.confusion_matrix(output, predicted_output)

        # Compute and return precision
        return float(confusion_matrix["true_positives"]) / float(confusion_matrix["true_positives"] +
                                                                 confusion_matrix["false_positives"])
