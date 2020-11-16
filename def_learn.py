import cv2
import numpy as np
import os
import def_filter as fi
import time

'''
skeleton image => oracle data (in txt)
'''

folderOutput = 'D:/oracleData/'


def imageRead(pathInput):
    image = cv2.imread(pathInput)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret_image_2, image_2 = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
    return image_2


def featureLearn(num):
    t1 = time.time()

    numStr = str(num)

    folderInput = 'D:/skeleton' + numStr + '/'

    imagelist = os.listdir(folderInput)

    for name in imagelist:
        path = folderInput + name

        # name without suffix
        realName = name[:-4]

        image_skl_ = imageRead(path)

        # oracle data => horizontal vertical slash backslash
        componentOf_ = fi.giveValueOfComponentOf_(image_skl_)
        componentOfI = fi.giveValueOfComponentOfI(image_skl_)
        componentOfSlash = fi.giveValueOfComponentOfSlash(image_skl_)
        componentOfBackslash = fi.giveValueOfComponentOfBacklash(image_skl_)

        dataForTxt = realName[:-2] + '.txt'
        strComponentOf_ = str(componentOf_) + ' '
        strComponentOfI = str(componentOfI) + ' '
        strComponentOfSlash = str(componentOfSlash) + ' '
        strComponentOfBackslash = str(componentOfBackslash) + ' '
        # output into txt
        f = open(folderOutput + dataForTxt, 'a')
        f.writelines([strComponentOf_, strComponentOfI, strComponentOfSlash, strComponentOfBackslash, '\n'])
        f.close()

        print(realName)
        print(componentOf_, componentOfI, componentOfSlash, componentOfBackslash)
        print('--------------------------')

        # oracle data => row mean value, column mean value, row variance, column variance
        image_skl_inv = 255 - image_skl_
        location = np.nonzero(image_skl_inv)
        rowLocation = location[0]
        columnLocation = location[1]
        # row and column mean
        meanOfRow = np.mean(rowLocation)
        meanOfColumn = np.mean(columnLocation)
        # row and column variance
        varOfRow = np.var(rowLocation)
        varOfColumn = np.var(columnLocation)

        nameOfTxt = 'location_' + realName[:-2] + '.txt'
        strMeanOfRow = str(meanOfRow) + ' '
        strMeanOfColumn = str(meanOfColumn) + ' '
        strVarOfRow = str(varOfRow) + ' '
        strVarOfColumn = str(varOfColumn) + ' '
        # output into txt
        f = open(folderOutput + nameOfTxt, 'a')
        f.writelines([strMeanOfRow, strMeanOfColumn, strVarOfRow, strVarOfColumn, '\n'])
        f.close()

        print(realName)
        print(meanOfRow, meanOfColumn, varOfRow, varOfColumn)
        print('--------------------------')

    t2 = time.time()
    t = t2 - t1
    print('time of ' + folderInput + ' is: ' + str(t))
