from knn import KNN
import cifar10

train_data, train_labels = cifar10.getTrainData()
test_data, test_labels = cifar10.getTestData()

# n son los n primeros del test_data
# k son los vecinos
n = 1000
k = 1

near = KNN(train_data, train_labels)
near.levenshtein(test_data[0:n],test_labels[0:n],k)
