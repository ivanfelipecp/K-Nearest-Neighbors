import numpy as np
import cifar10

# print(np.add(a1,a2))
# print(a2.min())
# data = unpickle("test_batch")

train_data, train_labels = cifar10.getTrainData()
print(train_data[0])
print(cifar10.getLabel(train_labels[0]))