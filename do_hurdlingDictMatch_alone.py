import cv2
import numpy as np
import os
import time
from collections import Counter
import pickle
import pprint
import def_hurdlingDictMatch as hDM
import gc
import sys
import def_similarCharacter as sC

'''
just alone implement prediction method hurdlingDictMatch 
'''

folderImage = 'D:/skeletonForHurdling/'


def giveAnser(listAnser, name):
    for it in range(len(listAnser)):
        if listAnser[it] == name:
            return True
    return False


t0 = time.time()
imageList = os.listdir(folderImage)
listAnswer_hurdlingDict = []
rightRate_hurdlingDict = []

for i in range(1, 33):
    t00 = time.time()

    listAnswer_hurdlingDict_tmp = []

    if i < 10:
        num = '0'+str(i)
    else:
        num = str(i)

    for imageName in imageList:
        if imageName[-6:-4] == num:
            image = cv2.imread(folderImage + imageName, cv2.IMREAD_GRAYSCALE)
            realName = imageName[:-6]
            answerName = hDM.matchBest_without(image, num+'/')
            if answerName == realName:
                listAnswer_hurdlingDict.append(1)
                listAnswer_hurdlingDict_tmp.append(1)
            elif sC.discriminateSimilarWord(answerName, realName):
                listAnswer_hurdlingDict.append(1)
                listAnswer_hurdlingDict_tmp.append(1)
            else:
                listAnswer_hurdlingDict.append(0)
                listAnswer_hurdlingDict_tmp.append(0)
            del image
            gc.collect()
        else:
            continue

    rightRate_hurdlingDict.append(listAnswer_hurdlingDict_tmp.count(1)/len(listAnswer_hurdlingDict_tmp))
    del listAnswer_hurdlingDict_tmp
    gc.collect()

    bar = i / 32
    print("finished: {:.2f} %".format(bar * 100))
    t11 = time.time()
    print('right rate: ')
    print(listAnswer_hurdlingDict.count(1) / len(listAnswer_hurdlingDict))
    print('rest time: ')
    print(str((32-i)*(t11-t00)/60)+'min')

t1 = time.time()
print('\n')
print('-------------------------')
print('summary time:')
print(str((t1-t0)/60)+' min')
print('right rate of hurdlingDict:')
print(listAnswer_hurdlingDict.count(1)/len(listAnswer_hurdlingDict))
print('best right rate:')
print(max(rightRate_hurdlingDict))
print('worst right rate:')
print(min(rightRate_hurdlingDict))

del listAnswer_hurdlingDict
del rightRate_hurdlingDict
gc.collect()


# result:
# summary time:
# 2646.6464834650355 min
# right rate of hurdlingDict：
# 0.9030979827089337
# best right rate:
# 0.9236311239193083
# worst right rate:
# 0.8645533141210374



