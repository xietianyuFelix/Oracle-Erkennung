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

'''
compute thresholds so that I can ensure 
that the list obtained by the hurdlingDict prediction method must contain the results I want.
'''

folderImage = 'D:/skeletonForHurdling/'

imageList = os.listdir(folderImage)

# threshold 1 for length of list
listOfDepth = []
# threshold 2 for match rate
listOfMatchDegree = []

t1 = time.time()

for i in range(1, 33):

    listOfDepth_tmp = []
    listOfMatchDegree_tmp = []

    k = i - 1
    print("progress bar{}{:.2f} %".format('>' * i, (k / 32) * 100))

    if i < 10:
        num = '0'+str(i)
    else:
        num = str(i)

    for imageName in imageList:
        if imageName[-6:-4] == num:   # extract the tail number of the sample  e.g. 01, 02,...,32
            image = cv2.imread(folderImage + imageName, cv2.IMREAD_GRAYSCALE)
            realName = imageName[:-6]
            threshold_depth, threshold_matchRate = hDM.giveBestThreshold_without(realName, image, num+'/')

            listOfDepth.append(threshold_depth)
            listOfMatchDegree.append(threshold_matchRate)

            listOfDepth_tmp.append(threshold_depth)
            listOfMatchDegree_tmp.append(threshold_matchRate)

            del image
            gc.collect()

        else:
            continue

    print('for image with tail number {0}, and find_depth is {1}'.format(num, max(listOfDepth_tmp)))
    print('for image with tail number {0}, and match rate is {1}'.format(num, min(listOfMatchDegree_tmp)))
    del listOfDepth_tmp
    del listOfMatchDegree_tmp
    gc.collect()

    t2 = time.time()
    tAlready = t2 - t1
    tSum = (t2-t1)/i*32
    tRest = tSum - tAlready
    print('rest time: {:.2f} min'.format(tRest/60))

t3 = time.time()

output1 = open('DepthList.pkl', 'wb')
pickle.dump(listOfDepth, output1)
output1.close()

output2 = open('MatchDegreeList.pkl', 'wb')
pickle.dump(listOfMatchDegree, output2)
output2.close()

bigestThreshold = max(listOfDepth)
smallestThreshold = min(listOfMatchDegree)

del listOfDepth
del listOfMatchDegree
gc.collect()

print('max find_depth: {0}'.format(bigestThreshold))
print('min match rate: {0}'.format(smallestThreshold))
print('time : ' + str((t3-t1)/60) + ' minuten')


# result:
# max find_depth: 316
# min match rate: 0.16981566820276398
# time : 2630 min
