import pandas as pd 
import numpy as np 
import string

def propername_featurize(input_data, n_grams, data_dict, label_dict, use_mask=True):
    """ Featurizes an input for the proper name domain.

    Inputs:
        input_data: The input data.
    """
    features = []

    for idx in range(input_data.shape[0]):
        feat = np.zeros(len(data_dict))

        label = input_data[idx,1]
        input_data[idx,0] = input_data[idx,0]#.lower()
        for n in n_grams:
            chars = [char for char in input_data[idx,0]]
            if use_mask and n > 2:
                chars = ['<MASK>'] + chars + ['<MASK>']

            for c in range(len(chars) - n):
                n_gram = ''.join(chars[c:c+n])
                if n_gram not in data_dict:
                    #If you don't want UNK, just continue
                    #continue
                    n_gram = '<UNK>'
                feat[data_dict[n_gram]] = 1
        
        features += [(feat.tolist(), label)]
    
    return features

def propername_data_loader(train_data_filename,
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
        Training, dev, and test data, all represented as (input, label) format.

        Suggested: for test data, put in some dummy value as the label.
    """

    USE_MASK = True
    N_GRAMS = [2,3]

    # TODO: Load the data from the text format.
    train_data = pd.read_csv(train_data_filename)
    train_labels = pd.read_csv(train_labels_filename)
    train = raw_from_df(train_data, train_labels, dummy_features=False)
    
    dev_data = pd.read_csv(dev_data_filename)
    dev_labels = pd.read_csv(dev_labels_filename)
    dev = raw_from_df(dev_data, dev_labels, dummy_features=False)
    
    test_data = pd.read_csv(test_data_filename)
    test = raw_from_df(test_data, None, dummy_features=True)

    # TODO: Featurize the input data for all three splits.
    data_dict, label_dict = create_dicts(train, N_GRAMS, use_mask=USE_MASK)

    print("data dict len: ", len(data_dict))

    train = propername_featurize(train, N_GRAMS, data_dict, label_dict, use_mask=USE_MASK)
    dev = propername_featurize(dev, N_GRAMS, data_dict, label_dict, use_mask=USE_MASK)
    test = propername_featurize(test, N_GRAMS, data_dict, label_dict, use_mask=USE_MASK)

    return train, dev, test, label_dict

def raw_from_df(data_df, label_df, dummy_features):
    feat_list = []

    for sample_id in data_df['id']:
        data = data_df.loc[data_df['id'] == sample_id]['text'].iloc[0]

        if dummy_features:
            label = 'dummy'
        else:
            label = label_df.loc[label_df['id'] == sample_id]['type'].iloc[0]

        feat_list.append(np.array((data, label)))

    return np.array(feat_list)

def create_dicts(input_data, n_grams, use_mask=True):
    """ Creates dicts of labels and n_grams

    Input:
        (list) list of n_grams to put into dict

    Returns:
        (dict) chars to feature index dict
        (dict) labels to output dict
    """

    #Initialize dicts with UNK tokens
    data_dict = { '<UNK>' : 0}
    label_dict = { '<UNK>' : 0}

    if use_mask:
        data_dict['<MASK>'] = 1

    for n_gram in n_grams:
        for row in input_data:
            text = row[0]#.lower()
            label = row[1]

            if label not in label_dict:
                label_dict[label] = len(label_dict)

            if (len(text) - n_gram) < 0:
                continue

            chars = [char for char in text]
            if use_mask and n_gram > 2:
                chars = ['<MASK>'] + chars + ['<MASK>']

            for c in range(len(chars) - n_gram):
                txt = ''.join(chars[c:c+n_gram])
                if txt not in data_dict:
                    data_dict[txt] = len(data_dict)

    return data_dict, label_dict



