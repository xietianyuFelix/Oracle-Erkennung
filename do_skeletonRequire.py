import time
import def_skeletonRequire


t1 = time.time()
for i in range(1, 140):
    def_skeletonRequire.skeletonImageRequire(i, True, True, True, 75)
t2 = time.time()
print((t2-t1)/60)

# t1 = time.time()
#
# def_skeletonRequire.skeletonImageRequire(139, True, True, False, 85)
#
# t2 = time.time()
# print((t2-t1)/60)


