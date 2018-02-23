from knn import KNN
import cifar10

train_data, train_labels = cifar10.getTrainData()
test_data, test_labels = cifar10.getTestData()

# n son los n primeros del test_data
# k son los vecinos
k = 3

near = KNN(train_data, train_labels)
near.levenstein(test_data,test_labels,1)
near.levenstein(test_data,test_labels,2)
near.levenstein(test_data,test_labels,3)
