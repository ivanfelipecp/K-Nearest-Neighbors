import numpy as np
import cPickle

dic_data = "data"
dic_labels = "labels"
labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def unpickle(f):
    with open(f,"rb") as fo:
        dictionary = cPickle.load(fo)
    # Keys of the dictionary = data, labels, filenames, batch_label
    return dictionary

def getTrainData(data_batch_filename="data_batch_", init_batch=1, end_batch=5):
    # Local and auxiliar variables
    cont = init_batch
    filename = data_batch_filename

    init_data = unpickle(filename+str(cont))
    train_data, train_labels = init_data[dic_data], init_data[dic_labels]

    # Starts in data_batch_2, ends in data_batch_5
    cont += 1
    while(cont <= end_batch):
        load_data = unpickle(filename+str(cont))
        train_data = np.concatenate((train_data,load_data[dic_data]),axis=0)
        train_labels = np.concatenate((train_labels,load_data[dic_labels]),axis=0)
        cont += 1
    return train_data, train_labels

def getTestData(test_batch_filename="test_batch"):
    init_data = unpickle(test_batch_filename)
    return init_data[dic_data], init_data[dic_labels]

def getLabel(n):
    return labels[n]