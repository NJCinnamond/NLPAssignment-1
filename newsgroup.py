def newsgroup_featurize(input_data):
    """ Featurizes an input for the newsgroup domain.

    Inputs:
        input_data: The input data.
    """
    # TODO: Implement featurization of input.
    pass

def newsgroup_data_loader(train_data_filename,
                          train_labels_filename,
                          dev_data_filename,
                          dev_labels_filename,
                          test_data_filename):
    """ Loads the data.

    Inputs:
        train_data_filename (str): The filename of the training data.
        train_labels_filename (str): The filename of the training labels.
        dev_data_filename (str): The filename of the development data.
        dev_labels_filename (str): The filename of the development labels.
        test_data_filename (str): The filename of the test data.

    Returns:
        Training, dev, and test data, all represented as a list of (input, label) format.

        Suggested: for test data, put in some dummy value as the label.
    """
    # TODO: Load the data from the text format.

    # TODO: Featurize the input data for all three splits.

    return [], [], []
