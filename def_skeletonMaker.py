import cv2
import numpy as np
import time

'''
reference:
http://cgm.cs.mcgill.ca/~godfried/teaching/projects97/azar/skeleton.html
(Hilditch's Algorithm)
https://blog.csdn.net/wsp_1138886114/article/details/101050825
(last part: black character white background)
'''


def Three_element_add(array_input):
    array0 = array_input[:]
    array1 = np.append(array_input[1:], np.array([0]))
    array2 = np.append(array_input[2:], np.array([0, 0]))
    arr_sum = array0 + array1 + array2
    return arr_sum[:-2]


def VThin(image_input, array_input):
    NEXT = 1
    height, width = image_input.shape[:2]
    for i in range(1, height):
        M_all = Three_element_add(image_input[i])
        for j in range(1, width):
            if NEXT == 0:
                NEXT = 1
            else:
                M = M_all[j - 1] if j < width - 1 else 1
                if image_input[i, j] == 0 and M != 0:
                    a = np.zeros(9)
                    if height - 1 > i and width - 1 > j:
                        kernel = image_input[i - 1:i + 2, j - 1:j + 2]
                        a = np.where(kernel == 255, 1, 0)
                        a = a.reshape(1, -1)[0]
                    NUM = np.array([1, 2, 4, 8, 0, 16, 32, 64, 128])
                    sumArr = np.sum(a * NUM)
                    image_input[i, j] = array_input[sumArr] * 255
                    if array_input[sumArr] == 1:
                        NEXT = 0
    return image_input


def HThin(image_input, array_input):
    height, width = image_input.shape[:2]
    NEXT = 1
    for j in range(1, width):
        M_all = Three_element_add(image_input[:, j])
        for i in range(1, height):
            if NEXT == 0:
                NEXT = 1
            else:
                M = M_all[i - 1] if i < height - 1 else 1
                if image_input[i, j] == 0 and M != 0:
                    a = np.zeros(9)
                    if height - 1 > i and width - 1 > j:
                        kernel = image_input[i - 1:i + 2, j - 1:j + 2]
                        a = np.where(kernel == 255, 1, 0)
                        a = a.reshape(1, -1)[0]
                    NUM = np.array([1, 2, 4, 8, 0, 16, 32, 64, 128])
                    sumArr = np.sum(a * NUM)
                    image_input[i, j] = array_input[sumArr] * 255
                    if array_input[sumArr] == 1:
                        NEXT = 0
    return image_input


def Xihua(binary_input, array_input, num=10):
    binary_image = binary_input.copy()
    image_output = cv2.copyMakeBorder(binary_image, 1, 0, 1, 0, cv2.BORDER_CONSTANT, value=0)
    for i in range(num):
        VThin(image_output, array_input)
        HThin(image_output, array_input)
    return image_output


def skeletonGive(binaryMatrix):
    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

    iThin = Xihua(binaryMatrix, array)
    return iThin
