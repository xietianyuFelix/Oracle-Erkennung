import numpy as np
import os
import time
import gc
import def_normalization as normalization

'''
notice:
you need normalization at first, then input to these functions

1 predict_knn_listAnswer
2 predict_knn_BestMatch_nearest
3 predict_clusterCenter_listAnswer  
4 predict_clusterCenter_BestMatch
'''


# name must input with '.txt'
def takeMatrix(name):
    listOfMat = []
    file = open(name, mode='r')
    for line in file:
        line = line.split()
        listOfMat.append(line)
    file.close()
    matrix = np.array(listOfMat)
    matrix2 = matrix.astype(float)
    del matrix
    del listOfMat
    gc.collect()
    return matrix2


def distaneceCalculate(centerM, otherM):
    return np.sqrt(np.sum(np.square(centerM - otherM)))


def clusterCenterGenerator(bigMatInput, listNameInput):
    centerBigMat = []
    centerNameList = []
    firstName = listNameInput[0]
    step = 1
    while listNameInput[step] == firstName:
        step = step+1
    for i in range(0, len(listNameInput), step):
        centerMat = np.mean(bigMatInput[i:i+step], axis=0)
        centerMat = np.reshape(centerMat, (1, -1))
        centerBigMat.append(centerMat)
        centerNameList.append(listNameInput[i])
    return np.concatenate(centerBigMat, axis=0), centerNameList


# mat1 = np.array([[1, 2, 1, 0, 0, 0],
#                  [2, 4, 1, 2, 0, 2],
#                  [4, 8, 1, 9, 0, 1],
#                  [3, 1, 1, 0, 0, 1],
#
#                  [0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0],
#
#                  [100, 2, 1, 0, 0, 0],
#                  [200, 4, 1, 2, 0, 2],
#                  [400, 8, 1, 9, 0, 1],
#                  [300, 1, 1, 0, 0, 1],
#
#                  [1, 2, 1, 0, 0, 0],
#                  [2, 4, 1, 2, 0, 2],
#                  [4, 8, 1, 9, 0, 1],
#                  [3, 1, 1, 0, 0, 1]])
# listName = ['1', '1', '1', '1', '2', '2', '2', '2', '3', '3', '3', '3', '4', '4', '4', '4']
# a, b = clusterCenterGenerator(mat1, listName)
# print(a)
# print(b)


def predict_knn_listAnswer(threshold, dataInput, bigMatInput, listNameInput):
    """
    accroding to threshold, return n names of nearest samples(distance smaller than threshold)
    output: a list of names with n sample points from near to far
    """
    dataInput = dataInput.reshape(1, -1)

    # test size
    hang, lie = dataInput.shape
    m, n = bigMatInput.shape
    lang = len(listNameInput)
    if lie != n:
        print('dataInput and bigMatInput not match')
        return None
    if m != lang:
        print('bigMatInput and listNameInput not match')
        return None

    holdDis = []
    holdName = []
    for i in range(m):
        dis = distaneceCalculate(dataInput, bigMatInput[i, :])
        if dis <= threshold:
            holdDis.append(dis)
            holdName.append(listNameInput[i])
    if len(holdName) == 0:
        print('threshold is too small')
        return None

    # Sort
    res = []
    index_small_big = np.argsort(np.array(holdDis))
    for k in index_small_big:
        res.append(holdName[k])

    del bigMatInput
    gc.collect()

    return res


# mat1 = np.array([[2, 2, 1],
#                  [2, 4, 1],
#                  [4, 4, 4],
#                  [1, 2, 2],
#                  [23, 33, 44],
#                  [44, 56, 87],
#                  [99, 99, 33]])
# listName = ['y', 'y1', 'y2', 'y_', 'cuo1', 'cuo2', 'cuo3']
# data = np.array([2, 2, 2])
# resout = predict_knn_listAnswer(15, data, mat1, listName)
# print(resout)


def predict_knn_BestMatch_nearest(dataInput, bigMatInput, listNameInput):
    """
    predict_knn_BestMatch_nearest：
    input: test data, normalized train data matrix, name list of train data matrix
    output: closest sample's name
    """
    dataInput = dataInput.reshape(1, -1)
    m, n = bigMatInput.shape
    distanceList = []
    nameList = []
    for i in range(m):
        dis = distaneceCalculate(dataInput, bigMatInput[i, :])
        distanceList.append(dis)
        nameList.append(listNameInput[i])
    return nameList[distanceList.index(min(distanceList))]


# mat1 = np.array([[2, 2, 1],
#                  [2, 4, 1],
#                  [4, 4, 4],
#                  [1, 2, 2],
#                  [23, 33, 44],
#                  [44, 56, 87],
#                  [99, 99, 33]])
# listName = ['y', 'y1', 'y2', 'y_', 'cuo1', 'cuo2', 'cuo3']
# data = np.array([2, 2, 2])
# resout = predict_knn_BestMatch_nearest(data, mat1, listName)
# print(resout)


def predict_clusterCenter_listAnswer(threshold, dataInput, centerMatInput, listNameInput):
    """
    predict_clusterCenter_listAnswer：
    accroding to threshold, return n names of nearest cluster center (distance smaller than threshold)
    output: a list of names with n cluster center from near to far
    """
    dataInput = dataInput.reshape(1, -1)

    # test size
    hang, lie = dataInput.shape
    m, n = centerMatInput.shape
    lang = len(listNameInput)
    if lie != n:
        print('dataInput and bigMatInput not match')
        return None
    if m != lang:
        print('centerMatInput and listNameInput not match')
        return None

    holdDis = []
    holdName = []
    for i in range(m):
        dis = distaneceCalculate(dataInput, centerMatInput[i, :])
        if dis <= threshold:
            holdDis.append(dis)
            holdName.append(listNameInput[i])
    if len(holdName) == 0:
        print('threshold too small')
        return None

    # sort
    res = []
    index_small_big = np.argsort(np.array(holdDis))
    for k in index_small_big:
        res.append(holdName[k])

    del centerMatInput
    gc.collect()

    return res


def predict_clusterCenter_BestMatch(dataInput, centerMatInput, listNameInput):
    """
    predict_clusterCenter_BestMatch：
      input: test data, normalized cluster center matrix of train data, name list of cluster center
      output: closest cluster center's name
    """
    dataInput = dataInput.reshape(1, -1)
    m, n = centerMatInput.shape

    distanceList = []
    nameList = []
    for i in range(m):
        dis = distaneceCalculate(dataInput, centerMatInput[i, :])
        distanceList.append(dis)
        nameList.append(listNameInput[i])
    return nameList[distanceList.index(min(distanceList))]
