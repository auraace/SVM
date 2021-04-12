
from libsvm.svmutil import *
import svmutil

def initializeWeights(m):
    W = []
    f = open("weights.txt", "w")
    for i in range(m):
        W.append(1/m)
        f.write(str(1/m))
        if i != m-1:
            f.write("\n")
    return W

def adaBoost(K):
    y, x = svmutil.svm_read_problem('DogsVsCats.train')
    W = initializeWeights(len(x))

    for t in range(1):
        m = svmutil.svm_train(y, x, '-t 0')


K = 10
adaBoost(K)