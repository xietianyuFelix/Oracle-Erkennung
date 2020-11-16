import numpy as np
import pickle
import os
import def_norm as nor
import gc
import time


'''
1 listForSimilarCharater_add(name1, name2):
  add similar names to similiar.pkl
2 discriminateSimilarWord(str1, str2):
  tell me whether they are a couple of similar names
'''

folder = 'D:/oracleData/'


def haveOrNot(listInput, name1, name2):
    if (name1 in listInput) & (name2 in listInput):
        return True
    elif name1 in listInput:
        return name1
    elif name2 in listInput:
        return name2
    else:
        return False


def saveDatslist(similarWordList):
    inputList = open('similiar.pkl', 'wb')
    pickle.dump(similarWordList, inputList)
    inputList.close()


def loadDataList(similarWordNamePKL):
    pkl_file = open(similarWordNamePKL, 'rb')
    dataListOut = pickle.load(pkl_file)
    pkl_file.close()
    return dataListOut


# add similar characters to similar.pkl
def listForSimilarCharater_add(name1, name2):
    if os.path.exists('similiar.pkl') is False:
        li = [name1, name2]
        dataList = [li]
        saveDatslist(dataList)
        print('initialization')
        return -1
    else:
        dataList = loadDataList('similiar.pkl')

        for i in range(len(dataList)):
            if haveOrNot(dataList[i], name1, name2) is True:
                print('already exists')
                return True
            elif haveOrNot(dataList[i], name1, name2) == name1:
                dataList[i].append(name2)
                print('already found '+name1+' in similiar.pkl and add '+name2+' in it')
                saveDatslist(dataList)
                return 1
            elif haveOrNot(dataList[i], name1, name2) == name2:
                dataList[i].append(name1)
                print('already found '+name2+' in similiar.pkl and add '+name1+' in it')
                saveDatslist(dataList)
                return 1
        li = [name1, name2]
        dataList.append(li)
        saveDatslist(dataList)
        print('similiar.pkl already updated')
        return 0


# Determine whether two names belong to
# the same similar character.
def discriminateSimilarWord(str1, str2):
    dataList = loadDataList('similiar.pkl')
    for i in range(len(dataList)):
        if str1 in dataList[i]:
            if str2 in dataList[i]:
                return True
            else:
                return False
    return False



