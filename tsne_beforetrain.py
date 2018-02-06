# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:14:29 2017

@author: Day
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from time import time
from skimage import io, color

'''
train1000=pd.read_csv('D:/pythoncode/tf1.0/Galaxy/data3/train1000_label.csv')
index1=train1000['GalaxyID']
for i in range(len(index1)):
    img=io.imread('D:/pythoncode/tf1.0/Galaxy/data3/train1000/'+ str(index1[i])+'.jpg') 
    img=img[106:318,106:318,:]
    io.imsave('D:/pythoncode/tf1.0/Galaxy/data3/train1000_212_212/'+str(index1[i])+'.jpg',img)
    
test1000=pd.read_csv('D:/pythoncode/tf1.0/Galaxy/data3/test1000_label.csv')
index1=test1000['GalaxyID']
for i in range(len(index1)):
    img=io.imread('D:/pythoncode/tf1.0/Galaxy/data3/test1000/'+ str(index1[i])+'.jpg') 
    img=img[106:318,106:318,:]
    io.imsave('D:/pythoncode/tf1.0/Galaxy/data3/test1000_212_212/'+str(index1[i])+'.jpg',img)
'''    


#train set tsne 
str2='D:/pythoncode/tf1.0/Galaxy/data3/train1000_212_212/*.jpg'
coll=io.ImageCollection(str2)
mat2=io.concatenate_images(coll)
mat2=mat2.reshape(-1,134832)  #212*212*3

train1000_label=pd.read_csv('D:/pythoncode/tf1.0/Galaxy/data3/train1000_label.csv')
train1000_label=train1000_label.sort_index(by='GalaxyID')
label_train1000=train1000_label['class'].as_matrix() 

print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
mat2_tsne = tsne.fit_transform(mat2)

x_min, x_max = np.min(mat2_tsne, 0), np.max(mat2_tsne, 0)
X = (mat2_tsne - x_min) / (x_max - x_min) #归一化
t1=time()
print("time  %.2fs" % (t1 - t0))

colors=['red','m','cyan','blue','lime']
plt.figure(figsize=(10,6))
for i in range(len(colors)):
    px=X[:,0][label_train1000==i]
    py=X[:,1][label_train1000==i]
    plt.scatter(px,py,c=colors[i],marker = 'o')
    
plt.legend(np.arange(0,5).astype(str))
plt.xticks([])
plt.yticks([])
plt.savefig('C:/Users/Day/Desktop/PPT_report/Galaxy pic/Visualization/2/train1000.png', dpi=300, bbox_inches='tight')
plt.show()


'''
#test set tsne 
str2='D:/pythoncode/tf1.0/Galaxy/data3/test1000_212_212/*.jpg'
coll=io.ImageCollection(str2)
mat2=io.concatenate_images(coll)
mat2=mat2.reshape(-1,134832)  #212*212*3

train1000_label=pd.read_csv('D:/pythoncode/tf1.0/Galaxy/data3/test1000_label.csv')
train1000_label=train1000_label.sort_index(by='GalaxyID')
label_train1000=train1000_label['class'].as_matrix() 

print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
mat2_tsne = tsne.fit_transform(mat2)

x_min, x_max = np.min(mat2_tsne, 0), np.max(mat2_tsne, 0)
X = (mat2_tsne - x_min) / (x_max - x_min) #归一化

t1=time()
print("time  %.2fs" % (t1 - t0))

colors=['red','m','cyan','blue','lime']
plt.figure(figsize=(10,6))
for i in range(len(colors)):
    px=X[:,0][label_train1000==i]
    py=X[:,1][label_train1000==i]
    plt.scatter(px,py,c=colors[i],marker = 'o')
    
plt.legend(np.arange(0,5).astype(str))
plt.xticks([])
plt.yticks([])
plt.savefig('C:/Users/Day/Desktop/PPT_report/Galaxy pic/Visualization/2/test1000.png', dpi=300, bbox_inches='tight')
plt.show()
'''