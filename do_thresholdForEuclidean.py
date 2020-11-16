import def_thresholdForEuclidean as thEuc
import pickle


thresholdList_clusterCenter = thEuc.giveThreshold_clusterCenter_listAnswer()

output1 = open('thresholdList_clusterCenter.pkl', 'wb')
pickle.dump(thresholdList_clusterCenter, output1)
output1.close()

thresholdList_knn = thEuc.giveThreshold_knn_listAnswer()

output2 = open('thresholdList_knn.pkl', 'wb')
pickle.dump(thresholdList_knn, output2)
output2.close()

