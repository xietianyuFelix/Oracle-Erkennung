import cv2
import numpy as np
import os
import time
from collections import Counter
import pickle
import pprint
import gc
import def_similarCharacter as sC


folderCode = 'D:/hurdlingDict/'
folderCodeBase = 'D:/hurdlingDict_without/'


# input a skeleton image, return an 1-D array,
# element of array is the number of pixels in each row or column
def hurdlingCode_image(image):
    image = image[4:-4, 4:-4]
    image = 255 - image
    image = image / 255
    code_h = np.sum(image, axis=1)
    code_v = np.sum(image, axis=0)
    code = np.hstack((code_h, code_v))
    del image
    gc.collect()
    return code


# comput match rate between new input image and a name of hurdlingDict code
# dataInput is HurdlingDict code in library
def matchRate(img, dataInput):
    rate = 0.0
    code_test = hurdlingCode_image(img)
    if len(code_test) != len(dataInput):
        print('size not match')
    for i in range(len(code_test)):
        key = code_test[i]
        rate = rate + (dataInput[i].get(key, 0.0))/32
    rate = rate / len(code_test)
    return rate


def matchRate_without(img, dataInput):
    rate = 0.0
    code_test = hurdlingCode_image(img)
    if len(code_test) != len(dataInput):
        print('size not match')
    for i in range(len(code_test)):
        key = code_test[i]
        rate = rate + (dataInput[i].get(key, 0.0))/31
    rate = rate / len(code_test)
    return rate


# >>> input a image, return best match result in folder or filterList
def matchBest(img, filterList=None):
    listRate = []
    listName = []
    codelist = os.listdir(folderCode)
    for codePath in codelist:
        if filterList is None or codePath[:-4] in filterList:
            pkl_file = open(folderCode + codePath, 'rb')
            data = pickle.load(pkl_file)
            pkl_file.close()
            rate = matchRate(img, data)
            listRate.append(rate)
            listName.append(codePath[:-4])
        else:
            continue
        del data
        gc.collect()
    # find best match
    indexMax = listRate.index(max(listRate))
    return listName[indexMax]


# >>> input a image, return best match result in folder or filterList
# for cross-validation
def matchBest_without(img, strNum_, filterList=None):
    listRate = []
    listName = []
    codelist = os.listdir(folderCodeBase+strNum_)
    for codePath in codelist:
        if filterList is None or codePath[:-4] in filterList:
            pkl_file = open(folderCodeBase + strNum_ + codePath, 'rb')
            data = pickle.load(pkl_file)
            pkl_file.close()
            rate = matchRate_without(img, data)
            listRate.append(rate)
            listName.append(codePath[:-4])
        else:
            continue
        del data
        gc.collect()
    indexMax = listRate.index(max(listRate))
    return listName[indexMax]


# input: listRate and listName   return: num length list  element: name with matching degree from high to low
# num is find depth, e.g. num=1 then depth find is 2, that is first and second largest value
# if num=0  only one result in list
def giveListAnswer(listR_input, listN_input, num):
    listR = listR_input.copy()
    listN = listN_input.copy()
    if num > len(listR):
        print('num is false')
        return None
    res = []
    for i in range(num+1):
        indexMaxMatch = listR.index(max(listR))
        res.append(listN[indexMaxMatch])
        del listR[indexMaxMatch]
        del listN[indexMaxMatch]
    return res


# li1 = [0.11, 0.44, 0.67, 0.66, 0.67, 0.99]
# li2 = ['smallest', 'smaller', 'big', 'middle', 'big', 'biggest']
# print(giveListAnswer(li1, li2, 1))


# >>> input:img, num (length of list)   return a list, the result that we wanted must be included
def MatchList(img, num):
    listRate = []
    listName = []
    codelist = os.listdir(folderCode)
    for codePath in codelist:
        pkl_file = open(folderCode + codePath, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        rate = matchRate(img, data)
        listRate.append(rate)
        listName.append(codePath[:-4])
        del data
        gc.collect()
    return giveListAnswer(listRate, listName, num)


# strNum_ =>  '01/'
def MatchList_without(img, num, strNum_):
    listRate = []
    listName = []
    codelist = os.listdir(folderCodeBase+strNum_)
    for codePath in codelist:
        pkl_file = open(folderCodeBase + strNum_ + codePath, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        rate = matchRate_without(img, data)
        listRate.append(rate)
        listName.append(codePath[:-4])
        del data
        gc.collect()
    return giveListAnswer(listRate, listName, num)


# >>> input:img, th (threshold of match rate)    return a list, the result that we wanted must be included
def MatchList_threshold(img, th):
    listResult = []
    codelist = os.listdir(folderCode)
    for codePath in codelist:
        pkl_file = open(folderCode + codePath, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        rate = matchRate(img, data)
        if rate > th:
            listResult.append(codePath[:-4])
        del data
        gc.collect()
    return listResult


# strNum_ =>  '01/'
def MatchList_threshold_without(img, th, strNum_):
    listResult = []
    codelist = os.listdir(folderCodeBase+strNum_)
    for codePath in codelist:
        pkl_file = open(folderCodeBase + strNum_ + codePath, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        rate = matchRate_without(img, data)
        if rate > th:
            listResult.append(codePath[:-4])
        del data
        gc.collect()
    return listResult


# input list of match rate, list of name, goal_name  return: find depth and corresponding match rate
def giveFindDepth(listR_input, listN_input, nameInput):
    listR = listR_input.copy()
    listN = listN_input.copy()
    depth = 0
    while len(listR) > 0:
        indexMatch = listR.index(max(listR))
        if nameInput == listN[indexMatch]:
            return depth, max(listR)
        elif sC.discriminateSimilarWord(nameInput, listN[indexMatch]):
            return depth, max(listR)
        else:
            del listR[indexMatch]
            del listN[indexMatch]
        depth = depth + 1
    print('nothing is found')
    return None


# li1 = [0.11, 0.44, 0.67, 0.66, 0.67, 0.99]
# li2 = ['smallest', 'smaller', 'big', 'middel', 'i_want_it', 'biggest']
# print(giveFindDepth(li1, li2, 'i_want_it'))
# print(li1)
# print(li2)
# print(giveListAnswer(li1, li2, 2))
# print(li1)
# print(li2)


# >>> for a image, return thresholds (find_depth and match rate)
# in other words, it can answer me, the thresholds for list so that the input name is included
def giveBestThreshold(realNameInput, img):
    listRate = []
    listName = []
    codelist = os.listdir(folderCode)
    for codePath in codelist:
        pkl_file = open(folderCode + codePath, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        rate = matchRate(img, data)
        listRate.append(rate)
        listName.append(codePath[:-4])
        del data
        gc.collect()
    return giveFindDepth(listRate, listName, realNameInput)


# strNum_ =>  '01/' for cross-validation
def giveBestThreshold_without(realNameInput, img, strNum_):
    listRate = []
    listName = []
    codelist = os.listdir(folderCodeBase+strNum_)
    for codePath in codelist:
        pkl_file = open(folderCodeBase+strNum_ + codePath, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        rate = matchRate_without(img, data)
        listRate.append(rate)
        listName.append(codePath[:-4])
        del data
        gc.collect()
    return giveFindDepth(listRate, listName, realNameInput)


