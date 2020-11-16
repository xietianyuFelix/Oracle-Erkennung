from sklearn.svm import LinearSVC
import time
import numpy as np
import pickle
import def_normalization as normalization
import def_similarCharacter as sC
import gc
import sys


folderTrain = 'D:/dataSet_crossValidation/01/'
folderTest = 'D:/testSet/01/'


def linear_SVC(trainSet, trainLabel, testData_input):
    cls = LinearSVC(dual=True)
    cls.fit(trainSet, trainLabel)
    return cls.predict(testData_input)


t1 = time.time()

bigMat_file = open(folderTrain + 'bigMat.pkl', 'rb')
bigMat_train = pickle.load(bigMat_file)
bigMat_file.close()

listName_file = open(folderTrain + 'listLabel.pkl', 'rb')
listName_train = pickle.load(listName_file)
listName_file.close()

bigMat_file_ = open(folderTest + 'bigMat.pkl', 'rb')
bigMat_test = pickle.load(bigMat_file_)
bigMat_file_.close()

listName_file_ = open(folderTest + 'listLabel.pkl', 'rb')
listName_test = pickle.load(listName_file_)
listName_file_.close()

list_right = []

for j in range(len(listName_test)):
    testData, trainData = normalization.normalizator(bigMat_test[j], bigMat_train)
    testData_ = np.reshape(testData, (1, -1))
    preName = linear_SVC(trainData, listName_train, testData_)
    if preName == listName_test[j] or sC.discriminateSimilarWord(listName_test[j], preName):
        list_right.append(1)
    else:
        list_right.append(0)

    sys.stdout.write("\rright rate {:.2f} %".format((list_right.count(1)/len(list_right)) * 100))
    sys.stdout.flush()

del bigMat_train
del listName_train
del bigMat_test
del listName_test
gc.collect()

t2 = time.time()
print('\n')
print(list_right.count(1)/len(list_right))
print(t2-t1)

# result
# 0.7809798270893372  (right rate)
# 12522.535569429398 (time)

