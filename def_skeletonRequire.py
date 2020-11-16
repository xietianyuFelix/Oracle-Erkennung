import cv2
import numpy as np
import def_skeletonMaker as skl
import os
import time


def imageRead(pathInput, th, blur_use):
    image = cv2.imread(pathInput)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur_use is True:
        gray = cv2.blur(gray, (5, 5))
    ret_image_2, image_2 = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    return image_2


# give frame (input image: white background black character)
def fangKuang(matrixInput, curse):
    matrixInput = 255 - matrixInput
    position4matrix = np.transpose(np.nonzero(matrixInput))
    rowMin = position4matrix[0][0]
    rowMax = position4matrix[-1][0]
    columnMin = np.min(position4matrix, axis=0)
    columnMax = np.max(position4matrix, axis=0)

    if curse:
        if rowMax - rowMin < 480:
            rowMedium = (rowMin + rowMax) // 2
            if rowMedium >= 240:
                rowMin = rowMedium - 240
                rowMax = rowMedium + 240
            else:
                rowMax = rowMin + 480

        if columnMax[1] - columnMin[1] < 360:
            columnMedium = (columnMax[1] + columnMin[1]) // 2
            if columnMedium >= 180:
                columnMin_ = columnMedium - 180
                columnMax_ = columnMedium + 180
            else:
                columnMin_ = columnMin[1]
                columnMax_ = columnMin_ + 360
            return rowMin, rowMax, columnMin_, columnMax_

    return rowMin, rowMax, columnMin[1], columnMax[1]


def blackAreaFind(matInput):
    m, n = matInput.shape
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            sumOfMat = np.sum(matInput[i - 1:i + 2, j - 1:j + 2])
            if sumOfMat == 0:
                return True
            else:
                continue
    return False


def makebetter(image, num):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i in range(num):
        image = cv2.erode(image, kernel)
    return image


def skeletonImageRequire(num, blur_use=False, more_blur_use=False, blackAreaEstimate=False, threshold=78, curse=False):
    numStr = str(num)

    sk_folder = 'D:/skeleton' + numStr + '/'
    train_folder = 'D:/trainData' + numStr + '/'

    imagelist = os.listdir(train_folder)

    # progress bar
    bar = 0
    if len(imagelist) == 0:
        speed = 0
        print(train_folder + 'nothing in folder')
    else:
        speed = 100 / len(imagelist)

    for name in imagelist:
        path = train_folder + name

        # Remove the suffix
        realName = name[:-4]

        img_2 = imageRead(path, threshold, blur_use)

        # eliminate burrs
        if more_blur_use is True:
            img_2 = cv2.blur(img_2, (5, 5))

        # get frame
        rMin, rMax, cMin, cMax = fangKuang(img_2, curse)

        # get small image matrix
        img_topFloor = img_2[rMin:(rMax + 1), cMin:(cMax + 1)]

        # resize
        img_small = cv2.resize(img_topFloor, (360, 480), interpolation=cv2.INTER_AREA)

        # get binary image again in oder to extraction of skeleton image
        ret_img_small_2, img_small_2 = cv2.threshold(img_small, 30, 255, cv2.THRESH_BINARY)

        # Zero padding
        img_small_02 = np.pad(img_small_2, ((4, 4), (4, 4)), 'constant', constant_values=(255, 255))

        # extract skeleton
        image_skl = skl.skeletonGive(img_small_02)
        img_skl_ = image_skl[1:, 1:]

        if blackAreaEstimate is True:
            index = 0
            while blackAreaFind(img_skl_) is True:
                img_skl2 = skl.skeletonGive(img_skl_)
                img_skl_ = img_skl2[1:, 1:]
                index = index + 1
                print(realName + ' black area finded:')
                print('times: ' + str(index))
        else:
            if blackAreaFind(img_skl_) is True:
                img_skl2 = skl.skeletonGive(img_skl_)
                img_skl_ = img_skl2[1:, 1:]
                print(realName + ' black area finded:')

        # write skeleton image into folder
        os.chdir(sk_folder)
        cv2.imwrite(realName + '.png', img_skl_)

        print(realName + ' finished')
        bar = bar + speed
        print('progress bar: ' + str(bar) + '%')

    print('>>>>' + train_folder + ' finished')


