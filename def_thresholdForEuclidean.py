import numpy as np
import os
import time
import gc
import def_predictEuclidean as pre
import pickle
import def_normalization as normalization


def distaneceCalculate(centerM, otherM):
    return np.sqrt(np.sum(np.square(centerM - otherM)))


def giveThreshold_knn_listAnswer():
    folderTrainBase = 'D:/dataSet_crossValidation/'
    folderTestBase = 'D:/testSet/'

    list_threshold = []

    t0 = time.time()
    for i in range(32):
        if i < 9:
            strNum = '0' + str(i + 1)
        else:
            strNum = str(i + 1)

        bigMat_file = open(folderTrainBase + '/' + strNum + '/' + 'bigMat.pkl', 'rb')
        bigMat_train = pickle.load(bigMat_file)
        bigMat_file.close()

        listName_file = open(folderTrainBase + '/' + strNum + '/' + 'listLabel.pkl', 'rb')
        listName_train = pickle.load(listName_file)
        listName_file.close()

        bigMat_file_ = open(folderTestBase + '/' + strNum + '/' + 'bigMat.pkl', 'rb')
        bigMat_test = pickle.load(bigMat_file_)
        bigMat_file_.close()

        listName_file_ = open(folderTestBase + '/' + strNum + '/' + 'listLabel.pkl', 'rb')
        listName_test = pickle.load(listName_file_)
        listName_file_.close()

        arrayName_train = np.array(listName_train)

        for j in range(len(listName_test)):
            disTestData = []
            testData, trainData = normalization.normalizator(bigMat_test[j], bigMat_train)
            index = np.argwhere(arrayName_train == listName_test[j])
            for k in index:
                kk = k[0]
                disTestData.append(distaneceCalculate(testData, trainData[kk]))

            list_threshold.append(min(disTestData))

            del testData
            del trainData
            gc.collect()

        del arrayName_train
        del bigMat_train
        del bigMat_test
        del listName_train
        del listName_test
        gc.collect()

        t1 = time.time()
        print(i)
        print('time: '+str(((t1 - t0) / 60)))
        print(max(list_threshold))
        print('----------------------')

    return list_threshold


# result:
# 0.43465968213793504


def giveThreshold_clusterCenter_listAnswer():
    folderTrainBase = 'D:/dataSet_crossValidation/'
    folderTestBase = 'D:/testSet/'

    list_threshold = []

    t0 = time.time()
    for i in range(32):
        if i < 9:
            strNum = '0' + str(i + 1)
        else:
            strNum = str(i + 1)

        bigMat_file = open(folderTrainBase + '/' + strNum + '/' + 'bigMat.pkl', 'rb')
        bigMat_train = pickle.load(bigMat_file)
        bigMat_file.close()

        listName_file = open(folderTrainBase + '/' + strNum + '/' + 'listLabel.pkl', 'rb')
        listName_train = pickle.load(listName_file)
        listName_file.close()

        bigMat_file_ = open(folderTestBase + '/' + strNum + '/' + 'bigMat.pkl', 'rb')
        bigMat_test = pickle.load(bigMat_file_)
        bigMat_file_.close()

        listName_file_ = open(folderTestBase + '/' + strNum + '/' + 'listLabel.pkl', 'rb')
        listName_test = pickle.load(listName_file_)
        listName_file_.close()

        for j in range(len(listName_test)):
            testData, trainData = normalization.normalizator(bigMat_test[j], bigMat_train)
            centerMat, centerNamelist = pre.clusterCenterGenerator(trainData, listName_train)
            index = centerNamelist.index(listName_test[j])
            list_threshold.append(distaneceCalculate(centerMat[index], testData))

            del testData
            del trainData
            del centerMat
            del centerNamelist
            gc.collect()
        del bigMat_train
        del bigMat_test
        del listName_train
        del listName_test
        gc.collect()

        t1 = time.time()
        print(i)
        print('time: ' + str(((t1 - t0) / 60)))
        print(max(list_threshold))
        print('----------------------')

    return list_threshold


# result
# 0.34953146399813456

