import numpy as np
import os
import def_norm as nor
import gc
import pickle
import time

folder = 'D:/oracleData/'

folderOutput = 'D:/dataSet/'
folderOutput_train = 'D:/dataSet_crossValidation/'
folderOutput_test = 'D:/testSet/'


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


def dsMaker():
    listMat = []
    listLabel = []
    for i in range(65, 91):
        headName = chr(i)  # i => 'A'~'Z'
        for m in range(20):
            for n in range(20):
                fileName = folder + headName + str(m + 1) + '_' + str(n + 1) + '_.txt'
                fileNameLocation = folder + 'location_' + headName + str(m + 1) + '_' + str(n + 1) + '_.txt'
                if os.path.exists(fileName):
                    matComponent = takeMatrix(fileName)
                    matLocation = takeMatrix(fileNameLocation)
                    mat8dimension = np.hstack((matComponent, matLocation))

                    listMat.append(mat8dimension)

                    h, w = mat8dimension.shape
                    for k in range(h):
                        listLabel.append(headName + str(m + 1) + '_' + str(n + 1) + '_')

                    del matComponent
                    del matLocation
                    del mat8dimension
                    gc.collect()

                else:
                    continue
    bigMat = np.concatenate(listMat, axis=0)
    # write data into folder
    output1 = open(folderOutput + 'bigMat.pkl', 'wb')
    pickle.dump(bigMat, output1)
    output1.close()
    output2 = open(folderOutput + 'listLabel.pkl', 'wb')
    pickle.dump(listLabel, output2)
    output2.close()

    del bigMat
    del listLabel
    gc.collect()


def dsMaker_crossValidator():    # deleteNumber 0-31
    bigMat_file = open(folderOutput + 'bigMat.pkl', 'rb')
    bigMat = pickle.load(bigMat_file)
    bigMat_file.close()
    listName_file = open(folderOutput + 'listLabel.pkl', 'rb')
    listName = pickle.load(listName_file)
    listName_file.close()
    for num in range(32):
        if num < 9:
            strNum = '0'+str(num+1)
        else:
            strNum = str(num+1)

        testSet_bigMat = bigMat[num::32]
        testSet_listName = listName[num::32]
        trainSet_bigMat = np.delete(bigMat, list(range(num, len(bigMat), 32)), axis=0)
        trainSet_listName = np.delete(listName, list(range(num, len(listName), 32)), axis=0)
        # write data into folder
        output_testBigMat = open(folderOutput_test + '/' + strNum + '/' + 'bigMat.pkl', 'wb')
        pickle.dump(testSet_bigMat, output_testBigMat)
        output_testBigMat.close()

        output_testListName = open(folderOutput_test + '/' + strNum + '/' + 'listLabel.pkl', 'wb')
        pickle.dump(testSet_listName, output_testListName)
        output_testListName.close()

        output_trainBigMat = open(folderOutput_train + '/' + strNum + '/' + 'bigMat.pkl', 'wb')
        pickle.dump(trainSet_bigMat, output_trainBigMat)
        output_trainBigMat.close()

        output_trainListName = open(folderOutput_train + '/' + strNum + '/' + 'listLabel.pkl', 'wb')
        pickle.dump(trainSet_listName, output_trainListName)
        output_trainListName.close()

        del testSet_bigMat
        del testSet_listName
        del trainSet_bigMat
        del trainSet_listName
        gc.collect()
    del bigMat
    del listName
    gc.collect()


dsMaker()
dsMaker_crossValidator()





