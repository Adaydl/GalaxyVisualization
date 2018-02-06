# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:59:41 2017

@author: Day
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from time import time

'''
#train1000
mat=pd.read_csv('D:/pythoncode/tf1.0/Galaxy/Visualization/Dieleman/train1000_2049.csv',header=None)
label1000=pd.read_csv('D:/pythoncode/tf1.0/Galaxy/Visualization/Dieleman/label_train1000.csv',header=None)

label1000=label1000.as_matrix()
mat=mat.as_matrix() 
pre_label1000=pd.read_csv('D:/pythoncode/tf1.0/Galaxy/Visualization/Dieleman/pre_label_train1000.csv',header=None)
pre_label1000=pre_label1000.as_matrix()
'''



#test1000
mat=pd.read_csv('D:/pythoncode/tf1.0/Galaxy/Visualization/Dieleman/test1000_2049.csv',header=None)
label1000=pd.read_csv('D:/pythoncode/tf1.0/Galaxy/Visualization/Dieleman/label_test1000.csv',header=None)

label1000=label1000.as_matrix()
mat=mat.as_matrix() 
pre_label1000=pd.read_csv('D:/pythoncode/tf1.0/Galaxy/Visualization/Dieleman/pre_label_test1000.csv',header=None)
pre_label1000=pre_label1000.as_matrix()


mat2=mat[:, 1:]
print("Computing t-SNE embedding")
#tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
mat2_tsne = tsne.fit_transform(mat2)



x_min, x_max = np.min(mat2_tsne, 0), np.max(mat2_tsne, 0)
X = (mat2_tsne - x_min) / (x_max - x_min) #归一化

t1=time()
print("time  %.2fs" % (t1 - t0))

mat3=np.column_stack((label1000,pre_label1000[:,1],X))
#np.savetxt('./train1000_5.csv', mat3, delimiter = ',') 
#np.savetxt('./test1000_5.csv', mat3, delimiter = ',')
 
label1000_2=label1000[:,1]
pre_label1000_2=pre_label1000[:,1]
colors=['red','m','cyan','blue','lime']

plt.figure(figsize=(10,6))

for i in range(len(colors)):
    px=[]
    py=[]
    px2=[]
    py2=[]
    for j in range(1000):
        if (label1000_2[j]==i and pre_label1000_2[j]==i):
            px.append(X[j,0])
            py.append(X[j,1])
           
        if (label1000_2[j]==i and pre_label1000_2[j]!=i):
            px2.append(X[j,0])
            py2.append(X[j,1])
           
    plt.scatter(px,py,s=20,c=colors[i],marker='o')
    plt.scatter(px2,py2,s=20,c=colors[i],marker='v')
    
#plt.legend(np.arange(0,5).astype(str))
plt.xticks([])
plt.yticks([])
#plt.savefig('C:/Users/Day/Desktop/PPT_report/Galaxy pic/Visualization/2/cnn1_train.png', dpi=300, bbox_inches='tight')
plt.savefig('C:/Users/Day/Desktop/PPT_report/Galaxy pic/Visualization/2/cnn1_test.png', dpi=300, bbox_inches='tight')

plt.show()    





