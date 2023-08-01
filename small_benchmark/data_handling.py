import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus'] = False	

loss_aftercorrect_list = np.load('/workspace/home/ubuntu/digest_yzy/pyg_autoscale/small_benchmark/2022_10_27_12_12_07/0_loss_aftercorrect_list.npy')
loss_precorrect_list = np.load('/workspace/home/ubuntu/digest_yzy/pyg_autoscale/small_benchmark/2022_10_27_12_12_07/0_loss_precorrect_list.npy')




xdata = np.arange(0, len(loss_aftercorrect_list))
print('Loss of predicted embedding:     ', 'Loss of historical embedding:    ', 'difference:')
for i in range(0, len(loss_aftercorrect_list), 10):
    print(loss_aftercorrect_list[i],'     ', loss_precorrect_list[i], '     ', loss_aftercorrect_list[i]-loss_precorrect_list[i])
# print('Loss of historical embedding:')
# for i in range(0, len(loss_precorrect_list), 10):
#     print(loss_precorrect_list[i])




plt.plot(xdata,loss_aftercorrect_list,color='r', ls='solid',marker='.',mec='r',mfc='w',label=u'predicted embedding')
plt.plot(xdata,loss_precorrect_list,color='b', ls='solid',marker='.',mec='b',mfc='w',label=u'historical embedding')     #color可自定义折线颜色，marker可自定义点形状，label为折线标注
plt.title(u"cora",size=10)
plt.legend()
plt.xlabel(u'epoch',size=10) 
plt.ylabel(u'Loss',size=10)
plt.savefig('image.png')

plt.show()
