import def_predictEuclidean as pre
import pickle
import gc
import time
import def_normalization as normalization
import def_hurdlingDictMatch as hDM
import cv2
import numpy as np
import def_similarCharacter as sC


folderTrainBase = 'D:/dataSet_crossValidation/'
folderTestBase = 'D:/testSet/'
folderImage = 'D:/skeletonForHurdling/'

th_knn = 0.4346
num_hurdling = 316
th_hurdling = 0.1698


def getImageMatFromName(nameInput):
    image = cv2.imread(folderImage + nameInput, cv2.IMREAD_GRAYSCALE)
    return image


def filterForTrainData(listSmall, listBig_, bigTrianData):
    listBig = listBig_.tolist()
    list_outputMat = []
    list_outputName = []
    for name in listSmall:
        indexStart = listBig.index(name)
        list_outputMat.append(bigTrianData[indexStart:indexStart+31])
        list_outputName.append(listBig_[indexStart:indexStart + 31])
    return np.concatenate(list_outputMat, axis=0), np.concatenate(list_outputName, axis=0)


def knn_hurdlingDict_without():
    """
    1 predict_knn_listAnswer
    2 hurdlingDict
    """
    list_knn_hD = []
    rightRate = []

    t0 = time.time()
    for i in range(32):
        list_knn_hD_tmp = []

        if i < 9:
            strNum = '0' + str(i + 1)
        else:
            strNum = str(i + 1)

        # take big mat and name list of train set
        bigMat_file = open(folderTrainBase + '/' + strNum + '/' + 'bigMat.pkl', 'rb')
        bigMat_train = pickle.load(bigMat_file)
        bigMat_file.close()

        listName_file = open(folderTrainBase + '/' + strNum + '/' + 'listLabel.pkl', 'rb')
        listName_train = pickle.load(listName_file)
        listName_file.close()

        # take big mat and name list of test set
        bigMat_file_ = open(folderTestBase + '/' + strNum + '/' + 'bigMat.pkl', 'rb')
        bigMat_test = pickle.load(bigMat_file_)
        bigMat_file_.close()

        listName_file_ = open(folderTestBase + '/' + strNum + '/' + 'listLabel.pkl', 'rb')
        listName_test = pickle.load(listName_file_)
        listName_file_.close()

        # predict each data in test set
        for j in range(len(listName_test)):
            # normalization
            testData, trainData = normalization.normalizator(bigMat_test[j], bigMat_train)
            # first predict method: knn_nearest => return a list of names
            preName_knn_list = pre.predict_knn_listAnswer(th_knn, testData, trainData, listName_train)
            # second predict method: hurdlingDictMatch => return prediction result
            img = getImageMatFromName(listName_test[j]+strNum+'.png')
            answer = hDM.matchBest_without(img, strNum+'/', preName_knn_list)

            if listName_test[j] == answer:
                list_knn_hD.append(1)
                list_knn_hD_tmp.append(1)
            elif sC.discriminateSimilarWord(listName_test[j], answer):
                list_knn_hD.append(1)
                list_knn_hD_tmp.append(1)
            else:
                list_knn_hD.append(0)
                list_knn_hD_tmp.append(0)

            del testData
            del trainData
            del img
            gc.collect()
        rightRate.append(list_knn_hD_tmp.count(1)/len(list_knn_hD_tmp))
        print('right rate for '+str(i)+' >>>'+str(list_knn_hD_tmp.count(1)/len(list_knn_hD_tmp)))

        del list_knn_hD_tmp
        del bigMat_train
        del bigMat_test
        del listName_train
        del listName_test
        gc.collect()

        t11 = time.time()
        t_res = ((t11 - t0) / (i + 1)) * (32 - (i + 1))
        print('rest time: ' + str(t_res / 60) + ' min')
        print('---------------')

    t1 = time.time()
    print('time: '+str((t1-t0)/60)+' min')
    print('right rate:')
    print(list_knn_hD.count(1)/len(list_knn_hD))
    print('best right rate:')
    print(max(rightRate))
    print('worst right rate:')
    print(min(rightRate))


def hurdlingDict_knn_without():
    """
    1 hurdlingDict
    2 predict_knn_nearest
    """
    list_hD_knn = []
    rightRate = []

    t0 = time.time()
    for i in range(32):

        list_hD_knn_tmp = []

        if i < 9:
            strNum = '0' + str(i + 1)
        else:
            strNum = str(i + 1)

        # take big mat and name list of train set
        bigMat_file = open(folderTrainBase + '/' + strNum + '/' + 'bigMat.pkl', 'rb')
        bigMat_train = pickle.load(bigMat_file)
        bigMat_file.close()

        listName_file = open(folderTrainBase + '/' + strNum + '/' + 'listLabel.pkl', 'rb')
        listName_train = pickle.load(listName_file)
        listName_file.close()

        # take big mat and name list of test set
        bigMat_file_ = open(folderTestBase + '/' + strNum + '/' + 'bigMat.pkl', 'rb')
        bigMat_test = pickle.load(bigMat_file_)
        bigMat_file_.close()

        listName_file_ = open(folderTestBase + '/' + strNum + '/' + 'listLabel.pkl', 'rb')
        listName_test = pickle.load(listName_file_)
        listName_file_.close()

        # predict each data in test set
        for j in range(len(listName_test)):
            # first predict method: hurdlingDictMatch => return a list of names
            img = getImageMatFromName(listName_test[j] + strNum + '.png')
            preName_hD_list = hDM.MatchList_without(img, num_hurdling, strNum + '/')
            # second predict method: knn_bestMatch_nearest => return prediction result
            trainData_hD, listName_train_hD = filterForTrainData(preName_hD_list, listName_train, bigMat_train)
            testData, trainData = normalization.normalizator(bigMat_test[j], trainData_hD)
            preName_nearst = pre.predict_knn_BestMatch_nearest(testData, trainData, listName_train_hD)
            if preName_nearst == listName_test[j]:
                list_hD_knn.append(1)
                list_hD_knn_tmp.append(1)
            elif sC.discriminateSimilarWord(preName_nearst, listName_test[j]):
                list_hD_knn.append(1)
                list_hD_knn_tmp.append(1)
            else:
                list_hD_knn.append(0)
                list_hD_knn_tmp.append(0)

            del img
            del preName_hD_list
            del trainData_hD
            del listName_train_hD
            del testData
            del trainData
            gc.collect()
        rightRate.append(list_hD_knn_tmp.count(1)/len(list_hD_knn_tmp))
        print('right rate for ' + str(i) + ' >>>' + str(list_hD_knn_tmp.count(1) / len(list_hD_knn_tmp)))

        del list_hD_knn_tmp
        del bigMat_train
        del bigMat_test
        del listName_train
        del listName_test
        gc.collect()

        t11 = time.time()
        t_res = ((t11 - t0) / (i + 1)) * (32 - (i + 1))
        print('rest time: ' + str(t_res / 60) + ' min')
        print('---------------')

    t1 = time.time()
    print('-------------------')
    print('time: ' + str((t1 - t0) / 60) + ' min')
    print('right rate:')
    print(list_hD_knn.count(1) / len(list_hD_knn))
    print('best right rate:')
    print(max(rightRate))
    print('worst right rate:')
    print(min(rightRate))


def hurdlingDict2_knn_without():
    """
    1 hurdlingDict
    2 predict_knn_nearest
    """
    list_hD_knn = []
    rightRate = []

    t0 = time.time()
    for i in range(32):

        list_hD_knn_tmp = []

        if i < 9:
            strNum = '0' + str(i + 1)
        else:
            strNum = str(i + 1)

        # take big mat and name list of train set
        bigMat_file = open(folderTrainBase + '/' + strNum + '/' + 'bigMat.pkl', 'rb')
        bigMat_train = pickle.load(bigMat_file)
        bigMat_file.close()

        listName_file = open(folderTrainBase + '/' + strNum + '/' + 'listLabel.pkl', 'rb')
        listName_train = pickle.load(listName_file)
        listName_file.close()

        # take big mat and name list of test set
        bigMat_file_ = open(folderTestBase + '/' + strNum + '/' + 'bigMat.pkl', 'rb')
        bigMat_test = pickle.load(bigMat_file_)
        bigMat_file_.close()

        listName_file_ = open(folderTestBase + '/' + strNum + '/' + 'listLabel.pkl', 'rb')
        listName_test = pickle.load(listName_file_)
        listName_file_.close()

        # predict each data in test set
        for j in range(len(listName_test)):
            # first predict method: hurdlingDictMatch => return a list of names
            img = getImageMatFromName(listName_test[j] + strNum + '.png')
            preName_hD_list = hDM.MatchList_threshold_without(img, th_hurdling, strNum + '/')
            # second predict method: knn_bestMatch_nearest => return prediction result
            trainData_hD, listName_train_hD = filterForTrainData(preName_hD_list, listName_train, bigMat_train)
            testData, trainData = normalization.normalizator(bigMat_test[j], trainData_hD)
            preName_nearst = pre.predict_knn_BestMatch_nearest(testData, trainData, listName_train_hD)
            if preName_nearst == listName_test[j]:
                list_hD_knn.append(1)
                list_hD_knn_tmp.append(1)
            elif sC.discriminateSimilarWord(preName_nearst, listName_test[j]):
                list_hD_knn.append(1)
                list_hD_knn_tmp.append(1)
            else:
                list_hD_knn.append(0)
                list_hD_knn_tmp.append(0)

            del img
            del preName_hD_list
            del trainData_hD
            del listName_train_hD
            del testData
            del trainData
            gc.collect()
        rightRate.append(list_hD_knn_tmp.count(1)/len(list_hD_knn_tmp))
        print('right rate for ' + str(i) + ' >>>' + str(list_hD_knn_tmp.count(1) / len(list_hD_knn_tmp)))

        del list_hD_knn_tmp
        del bigMat_train
        del bigMat_test
        del listName_train
        del listName_test
        gc.collect()

        t11 = time.time()
        t_res = ((t11-t0)/(i+1))*(32-(i+1))
        print('rest time: '+str(t_res/60)+' min')
        print('---------------')

    t1 = time.time()
    print('-------------------')
    print('time: ' + str((t1 - t0) / 60) + ' min')
    print('right rate:')
    print(list_hD_knn.count(1) / len(list_hD_knn))
    print('best right rate:')
    print(max(rightRate))
    print('worst right rate:')
    print(min(rightRate))


print('result of knn_hurdlingDict:')
knn_hurdlingDict_without()
print('======================')
print('result of hurdlingDict_knn:')
hurdlingDict_knn_without()
print('======================')
print('result of hurdlingDict2_knn:')
hurdlingDict2_knn_without()


# result of knn_hurdlingDict:
# time: 1385.6870011727015 min
# right rate:
# 0.9258375360230547
# best right rate:
# 0.9438040345821326
# worst right rate:
# 0.899135446685879
# ======================
# result of hurdlingDict_knn:
# time: 2394.9434847434363 min
# right rate:
# 0.9653278097982709
# best right rate:
# 0.9855907780979827
# worst right rate:
# 0.9438040345821326
# ======================
# result of hurdlingDict2_knn:
# time: 2335.1866877794264 min
# right rate:
# 0.9632564841498559
# best right rate:
# 0.9798270893371758
# worst right rate:
# 0.9409221902017291

