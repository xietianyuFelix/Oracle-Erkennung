import cv2
import numpy as np
import os
import time
from collections import Counter
import pickle
import pprint
import gc
import sys

'''
for each character make a list => HurdlingDict code
each element of list is a dic, which computed by a line of all samples
key of dict: passible number of pixels in that line
value of dict: probabiity of this passible number in all samples
'''

folderOutput = 'D:/hurdlingDict/'
folderOutput_without = 'D:/hurdlingDict_without/'
folderInput = 'D:/skeletonForHurdling/'


# input a skeleton image, return an 1-D array,
# element of array is the number of pixels in each row or column
def hurdlingCode_image(image):
    image = image[4:-4, 4:-4]   # Cut off the edges
    image = 255 - image
    image = image / 255
    code_h = np.sum(image, axis=1)
    code_v = np.sum(image, axis=0)
    code = np.hstack((code_h, code_v))
    del image
    gc.collect()
    return code

# print(type(hurdlingCode_image(cv2.imread('testImage.png', cv2.IMREAD_GRAYSCALE))))
# print(hurdlingCode_image(cv2.imread('testImage.png', cv2.IMREAD_GRAYSCALE)).shape)
# (840,)


# input a array, return dict (frequency of elements in array)
def dictMaker(arrayInput):
    return dict(Counter(arrayInput.flatten()))


# input real name like A1_1_, find all samples for it, return a matrix
# matrix row is 32 => 32 samples, column is 840 => 840 elements of HurdlingCode
def codeMat(name):
    listMat = []
    for num in range(1, 33):
        if num < 10:
            path = folderInput + name + '0' + str(num) + '.png'
        else:
            path = folderInput + name + str(num) + '.png'
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            listMat.append(np.reshape(hurdlingCode_image(img), (1, -1)))
            del img
            gc.collect()
        else:
            continue
    bigMat = np.concatenate(listMat, axis=0)
    return bigMat

# a = codeMat('A1_1_')
# print(a.shape)
# # (32, 840)
# print(a[:, 0].shape)
# # (32,)


# Ignore the number sample image (number: 1~32)
def codeMat_withoutNum(name, number):
    listMat = []
    for num in range(1, 33):
        if num == number:
            continue
        if num < 10:
            path = folderInput + name + '0' + str(num) + '.png'
        else:
            path = folderInput + name + str(num) + '.png'
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            listMat.append(np.reshape(hurdlingCode_image(img), (1, -1)))
            del img
            gc.collect()
        else:
            continue
    bigMat = np.concatenate(listMat, axis=0)
    return bigMat


# input real name like A1_1_, return a list for it => HurdlingDict code
# Dictionary statistics for each column of bigMat
def list_dict_word(name):
    matForWord = codeMat(name)
    res_list = []
    m, n = matForWord.shape
    for lie in range(n):
        res_list.append(dictMaker(matForWord[:, lie]))
    del matForWord
    gc.collect()
    return res_list


def list_dict_word_withoutNum(name, number):
    matForWord = codeMat_withoutNum(name, number)
    res_list = []
    m, n = matForWord.shape
    for lie in range(n):
        res_list.append(dictMaker(matForWord[:, lie]))
    del matForWord
    gc.collect()
    return res_list


# write the result(HurdlingDict code) in folder
def writeDictinFolder():
    imgList = os.listdir(folderInput)
    totalImage = len(imgList)   # for time prediction
    indexFinish = 0
    if totalImage == 0:
        print('no image in input folder')
        return None
    for i in range(65, 91):      # i = 'A' ~ 'Z'
        headName = chr(i)
        for m in range(21):
            for n in range(11):
                namePath = folderInput + headName + str(m) + '_' + str(n) + '_'
                nameWord = headName + str(m) + '_' + str(n) + '_'
                if os.path.exists(namePath+'01.png'):
                    if os.path.exists(folderOutput+nameWord+'.pkl'):
                        pass
                    else:
                        listForWord = list_dict_word(nameWord)
                        output = open(folderOutput + nameWord + '.pkl', 'wb')
                        pickle.dump(listForWord, output)
                        output.close()
                        del listForWord
                        gc.collect()
                    # for time predict
                    indexFinish = indexFinish + 32
                    bar = indexFinish / totalImage
                    sys.stdout.write("\rfinished {:.2f} %".format(bar * 100))
                    sys.stdout.flush()
                else:
                    continue
    print('\n')
    print('write is finished')


# Ignore the number sample image (number: 1~32)
def writeDictinFolder_withoutNum(number):
    if number < 10:
        strNum = '0'+str(number)
    else:
        strNum = str(number)

    imgList = os.listdir(folderInput)
    totalImage = len(imgList)
    indexFinish = 0
    if totalImage == 0:
        print('no image in input folder')
        return None
    for i in range(65, 91):
        headName = chr(i)
        for m in range(21):
            for n in range(11):
                namePath = folderInput + headName + str(m) + '_' + str(n) + '_'
                nameWord = headName + str(m) + '_' + str(n) + '_'
                if os.path.exists(namePath+'01.png'):
                    if os.path.exists(folderOutput_without+strNum+'/'+nameWord+'.pkl'):
                        pass
                    else:
                        listForWord = list_dict_word_withoutNum(nameWord, number)
                        output = open(folderOutput_without+strNum+'/'+nameWord + '.pkl', 'wb')
                        pickle.dump(listForWord, output)
                        output.close()
                        del listForWord
                        gc.collect()
                    indexFinish = indexFinish + 32
                    bar = indexFinish / totalImage
                    sys.stdout.write("\rfinished {:.2f} %".format(bar*100))
                    sys.stdout.flush()
                else:
                    continue
    print('\n')
    print('write is finished')



