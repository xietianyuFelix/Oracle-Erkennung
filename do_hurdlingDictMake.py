import def_hurdlingDict as hD


hD.writeDictinFolder()

for i in range(1, 33):
    hD.writeDictinFolder_withoutNum(i)
    print('finished: '+str((i/32)*100)+'%')
