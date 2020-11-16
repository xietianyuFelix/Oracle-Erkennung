import time
import def_learn


t1 = time.time()
for i in range(1, 140):
    def_learn.featureLearn(i)
t2 = time.time()
print('time: ')
print((t2-t1)/60)
