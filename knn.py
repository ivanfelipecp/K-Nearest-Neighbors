from __future__ import division
import numpy as np

class KNN:
    def __init__(self,train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.type = np.dtype([("image",np.float64), ("label", np.str_,16)])

    def sortAndClassify(self,distances,k):        
        # Instanciamos la estructura
        result = np.array(distances, dtype=self.type)

        # La sortiamos por img, de menor a mayor diferencia
        result = np.sort(result, order="image")

        # Extraemos los k vecinos
        neighbors = []
        for i in xrange(k):
            neighbors.append(result[i][1])
        
        # Retornamos el que mas aparece
        return np.bincount(neighbors).argmax()

    def chevyshev(self,test_data, test_labels, k):
        n = len(test_data)
        m = len(self.train_data)
        cont = 0

        print("*** Iniciando clasificacion de chevyshev con k=" + str(k) + " ***")
        for i in xrange(n):
            distances = []
            for j in xrange(m):
                dist = np.amax(np.absolute(np.subtract(self.train_data[j],test_data[i])))
                distances.append((dist,self.train_labels[j]))
            result = self.sortAndClassify(distances, k)
            if(result == test_labels[i]):
                cont += 1
            #print("resultado: "+str(result) + " | deberia ser: " + str(test_labels[i]))
        
        mul = cont/len(test_data) * 100 #float(cont * 1.0 /len(test_data * 1.0))
        print "El porcentaje de acierto fue {0}%".format(mul)
    
    def euclidean(self,test_data, test_labels, k):
        # Variables
        n = len(test_data)
        m = len(self.train_data)
        cont = 0

        print("*** Iniciando clasificacion euclideana con k=" + str(k) + " ***")
        for i in xrange(n):
            distances = []
            for j in xrange(m):
                dist = np.sqrt(np.sum(np.power(np.subtract(self.train_data[j],test_data[i]),2)))
                distances.append((dist,self.train_labels[j]))
            result = self.sortAndClassify(distances, k)
            if(result == test_labels[i]):
                cont += 1
            #print("resultado: "+str(result) + " | deberia ser: " + str(test_labels[i]))
        
        mul = cont/len(test_data) * 100 #float(cont * 1.0 /len(test_data * 1.0))
        print "El porcentaje de acierto fue {0}%".format(mul) 

    def manhattan(self, test_data, test_labels, k):
        n = len(test_data)
        m = len(self.train_data)
        cont = 0

        print("*** Iniciando clasificacion manhattan con k=" + str(k) + " ***")
        for i in xrange(n):
            distances = []
            for j in xrange(m):
                dist = np.sum(np.absolute(np.subtract(self.train_data[j],test_data[i])))
                distances.append((dist,self.train_labels[j]))
            result = self.sortAndClassify(distances, k)
            if(result == test_labels[i]):
                cont += 1
            #print("resultado: "+str(result) + " | deberia ser: " + str(test_labels[i]))
        
        mul = cont/len(test_data) * 100 #float(cont * 1.0 /len(test_data * 1.0))
        print "El porcentaje de acierto fue {0}%".format(mul)


    def levenstein(self, test_data, test_labels, k):
        n = len(test_data)
        m = len(self.train_data)
        cont = 0

        print("*** Iniciando clasificacion levenstein con k=" + str(k) + " ***")
        for i in xrange(n):
            distances = []
            for j in xrange(m):
                dist = np.sum(np.not_equal(self.train_data[j],test_data[i]))#np.sum(np.absolute(np.subtract(self.train_data[j],test_data[i])))
                distances.append((dist,self.train_labels[j]))
            result = self.sortAndClassify(distances, k)
            if(result == test_labels[i]):
                cont += 1
            #print("resultado: "+str(result) + " | deberia ser: " + str(test_labels[i]))
        
        mul = cont/len(test_data) * 100 #float(cont * 1.0 /len(test_data * 1.0))
        print "El porcentaje de acierto fue {0}%".format(mul)

    