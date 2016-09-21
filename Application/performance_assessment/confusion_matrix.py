import pandas as pd


class ConfusionMatrix:
    """Class to compute confusion matrix.

    Computes the confusion matrix for classification algorithms. Furthermore, includes functions to compute recall
    and precision.

    """

    @staticmethod
    def confusion_matrix(output, predicted_output):
        """Computes confusion matrix.

        Produces a confusion matrix.
        Where,
            True Label: +1, Predicted Label: +1 = True Positive
            True Label: +1, Predicted Label: -1 = False Negative
            True Label: -1, Predicted Label: +1 = False Positive
            True Label: -1, Predicted Label: -1 = True Negative

        Args:
            output (numpy.array): A numpy array containing real output.
            predicted_output (numpy.array): A numpy array containing predicted output.

        Returns:
            dict: A dictionary that contains the confusion matrix,
                {
                    'true_positives' (int): True positives count,
                    'false_negatives' (int): False negatives count,
                    'false_positives' (int): False positives count,
                    'true_negatives' (int): True negatives count
                }

        """
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
        """Computes precision.

        Computes the precision of the output and predicted_output.
        Precision =         # True positives
                   ------------------------------------
                   # True positives + # False negatives

        Args:
            output (numpy.array): A numpy array containing real output.
            predicted_output (numpy.array): A numpy array containing predicted output.

        Returns:
            float: Precision.

        """
        # Get the confusion matrix
        confusion_matrix = self.confusion_matrix(output, predicted_output)

        # Compute and return precision
        return float(confusion_matrix["true_positives"])/float(confusion_matrix["true_positives"] +
                                                               confusion_matrix["false_negatives"])

    def recall(self, output, predicted_output):
        """Computes recall.

        Computes the recall of the output and predicted output.
        Recall =        # True positives
               ------------------------------------
               # True positives + # False positives

        Args:
            output           (numpy.array): A numpy array containing real output.
            predicted_output (numpy.array): A numpy array containing predicted output.

        Returns:
            float: Recall.

        """
        # Get the confusion matrix
        confusion_matrix = self.confusion_matrix(output, predicted_output)

        # Compute and return precision
        return float(confusion_matrix["true_positives"]) / float(confusion_matrix["true_positives"] +
                                                                 confusion_matrix["false_positives"])
