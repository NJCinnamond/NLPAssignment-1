""" Maximum entropy model for Assignment 1: Starter code.

You can change this code however you like. This is just for inspiration.

"""
import os
import sys
import numpy as np

from util import evaluate, load_data
from sklearn.metrics import confusion_matrix

class PerceptronModel():
    """ Maximum entropy model for classification.

    Attributes:
    (float) weights
    (float) bias
    (int) num_dim
    (bool) add_bias

    """
    def __init__(self, label_to_index, lr=0.02):
        self.W = None
        self.bias = None
        self.lr = lr
        self.num_dim = 0
        self.num_class = len(label_to_index)
        self.label_to_index = label_to_index
        self.index_to_label = {v: k for k, v in label_to_index.items()}

    def train(self, training_data):
        """ Trains the maximum entropy model.

        Inputs:
            training_data: Suggested type is (list of pair), where each item is
                a training example represented as an (input, label) pair.
        """

        self.num_dim = len(training_data[0][0])
        self.num_epochs = 5
        self.W = {c: np.array([0.0 for _ in range(self.num_dim)]) for c in self.label_to_index.keys()}

        epoch = 0
        change_over_epoch = True
        while change_over_epoch and epoch < self.num_epochs:
            print("Epoch: ", epoch)
            epoch += 1
            correct = 0
            change_over_epoch = False

            for sample in training_data:
                #Get numerical value of label
                label = sample[1]
                if sample[1] not in self.label_to_index.keys():
                    label = self.index_to_label[0]

                # Initialize arg_max value, predicted class.
                arg_max, predicted_label = 0, self.index_to_label[0]

                # Multi-Class Decision Rule:
                for c in self.label_to_index.keys():
                    current_activation = np.dot(sample[0], self.W[c])
                    if current_activation >= arg_max:
                        arg_max, predicted_label = current_activation, c

                # Update Rule:
                if not (label == predicted_label):
                    change_over_epoch = True
                    self.W[label] += np.dot(self.lr, sample[0])
                    self.W[predicted_label] -= np.dot(self.lr, sample[0])
                else:
                    correct += 1
            
            acc = correct / len(training_data)
            print("Accuracy: ", str(acc))

    def predict(self, model_input):
        """ Predicts a label for an input.

        Inputs:
            model_input (features): Input data for an example, represented as a
                feature vector.

        Returns:
            The predicted class.    

        """
        # Initialize predicted label to UNK token
        arg_max, predicted_label = 0, self.index_to_label[0]

        # Multi-Class Decision Rule:
        for c in self.label_to_index.keys():
            current_activation = np.dot(model_input, self.W[c])
            if current_activation >= arg_max:
                arg_max, predicted_label = current_activation, c

        return predicted_label

def create_dummy_bias(data):
    for sample in data:
        sample[0].append(1)
    return data 

if __name__ == "__main__":
    print("Getting data")
    train_data, dev_data, test_data, data_type, label_dict = load_data(sys.argv)
    print("Got data")
    train_data = create_dummy_bias(train_data)
    dev_data = create_dummy_bias(dev_data)
    test_data = create_dummy_bias(test_data)

    print(len(train_data))
    print(len(dev_data))
    print(len(test_data))

    # Train the model using the training data.
    model = PerceptronModel(label_to_index=label_dict)
    model.train(train_data)

    # Predict on the development set. 
    '''
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", "perceptron_" + data_type + "_dev_predictions.csv"))
    print("Dev accuracy: ", dev_accuracy)
    '''
    pred_label = [model.predict(example[0]) for example in dev_data]
    true_label = [example[1] for example in dev_data]
    conf_mat = confusion_matrix(true_label, pred_label, 
            labels=np.sort(np.unique(true_label)))
    print(conf_mat)
    print(np.sort(np.unique(true_label)))
    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    #evaluate(model,
    #         test_data,
    #         os.path.join("results", "perceptron_" + data_type + "_test_predictions.csv"))
