import numpy as np
import os
import time
import gc
import def_predictEuclidean as pre
import pickle
import def_normalization as normalization
import sys
import def_similarCharacter as sC

'''
just alone implement prediction method knn_BestMatch_nearest
just alone implement prediction method predict_clusterCenter_BestMatch
'''

folderTrainBase = 'D:/dataSet_crossValidation/'
folderTestBase = 'D:/testSet/'

t0 = time.time()

listAnswer_nearst = []
listAnswer_cluster = []
rightRate_nearst = []
rightRate_cluster = []

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

    listAnswer_nearst_tmp = []
    listAnswer_cluster_tmp = []

    for j in range(len(listName_test)):
        testData, trainData = normalization.normalizator(bigMat_test[j], bigMat_train)

        preName_nearst = pre.predict_knn_BestMatch_nearest(testData, trainData, listName_train)
        if listName_test[j] == preName_nearst:
            listAnswer_nearst.append(1)
            listAnswer_nearst_tmp.append(1)
        elif sC.discriminateSimilarWord(listName_test[j], preName_nearst):
            listAnswer_nearst.append(1)
            listAnswer_nearst_tmp.append(1)
        else:
            listAnswer_nearst.append(0)
            listAnswer_nearst_tmp.append(0)

        centerMat, centerNamelist = pre.clusterCenterGenerator(trainData, listName_train)
        preName_cluster = pre.predict_clusterCenter_BestMatch(testData, centerMat, centerNamelist)
        if listName_test[j] == preName_cluster:
            listAnswer_cluster.append(1)
            listAnswer_cluster_tmp.append(1)
        elif sC.discriminateSimilarWord(listName_test[j], preName_cluster):
            listAnswer_cluster.append(1)
            listAnswer_cluster_tmp.append(1)
        else:
            listAnswer_cluster.append(0)
            listAnswer_cluster_tmp.append(0)

        del testData
        del trainData
        gc.collect()

    rightRate_nearst.append(listAnswer_nearst_tmp.count(1)/len(listAnswer_nearst_tmp))
    rightRate_cluster.append(listAnswer_cluster_tmp.count(1)/len(listAnswer_cluster_tmp))

    del listAnswer_nearst_tmp
    del listAnswer_cluster_tmp
    del bigMat_train
    del bigMat_test
    del listName_train
    del listName_test
    gc.collect()

    # progress bar
    bar = (i+1) / 32
    sys.stdout.write("\rfinished {:.2f} %".format(bar * 100))
    sys.stdout.flush()

print('\n')
t1 = time.time()
print('time: '+str((t1 - t0)/60)+' min')
print('The right Rate of knn_BestMatch_nearest:')
print((listAnswer_nearst.count(1)) / len(listAnswer_nearst))
print('best right rate:')
print(max(rightRate_nearst))
print('worst right rate:')
print(min(rightRate_nearst))
print('--------------')
print('The right Rate of clusterCenter_BestMatch:')
print((listAnswer_cluster.count(1)) / len(listAnswer_cluster))
print('best right rate:')
print(max(rightRate_cluster))
print('worst right rate:')
print(min(rightRate_cluster))

del listAnswer_nearst
del listAnswer_cluster
del rightRate_nearst
del rightRate_cluster
gc.collect()


# result:
# time: 81.40478842258453 min
# The right Rate of knn_BestMatch_nearest:
# 0.9535302593659942
# best right rate:
# 0.9726224783861671
# worst right rate:
# 0.9322766570605188
# --------------
# The right Rate of clusterCenter_BestMatch:
# 0.9232708933717579
# best right rate:
# 0.9481268011527377
# worst right rate:
# 0.8919308357348703



