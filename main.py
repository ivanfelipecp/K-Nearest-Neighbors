from knn import KNN
import cifar10

train_data, train_labels = cifar10.getTrainData()
test_data, test_labels = cifar10.getTestData()

# n son los n primeros del test_data
# k son los vecinos
k = 3

near = KNN(train_data, train_labels)
print("k -> 3")
print("chevychev")
near.chevyshev(test_data,test_labels,k)
print("manhattan")
near.manhattan(test_data,test_labels,k)
