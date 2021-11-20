# -*- coding: utf-8 -*-

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

'''
SMOTE Paper Implementation: https://arxiv.org/pdf/1106.1813.pdf
'''

'''
  Get k nearest neighbours for minority samples
'''

def get_knn(X, y, y_minority, k):
  neighbours = []
  for index_1 in range(len(X[:][y==y_minority])):
    neighbours.append([])
    for index_2 in range(len(X[:][y==y_minority])):
      if index_2 != index_1:
        dist = np.sum(np.square(X[:][y==y_minority][index_1]-X[:][y==y_minority][index_2]))
        neighbours[index_1].append((dist, index_2))
    neighbours[index_1].sort()
  neighbours = np.array(neighbours)
  k_neighbours = neighbours[:, :k]
  return k_neighbours


'''
  NN vector selection for oversampling
'''

def nn_vect_select(X, y, y_minority, nnarray, k, os_percent):
  init_size = len(X[:][y==y_minority])
  factor = os_percent // 100
  for index_1 in range(init_size):
    neigh_nums = random.sample(range(0, k), factor)
    for i in neigh_nums:
      index_2 = int(nnarray[index_1][i][1])
      ratio = random.random()
      new_index = X[:][y==y_minority][index_1] + ratio * (X[:][y==y_minority][index_2] - X[:][y==y_minority][index_1])
      X = np.append(X,[new_index],axis=0)
      y = np.append(y,[y_minority])
  return X, y


if __name__ == "__main__":
  num_samples = int(input("Number of samples? "))                                  # number of samples
  num_features = int(input("Number of features? "))                                # number of features
  num_classes = 2                                                                  # number of classes

  nok = 1
  weights_list = []
  while nok:
    weights_list = [float(item) for item in input("List of weights of the 2 classes? ").split()]
    if len(weights_list) == num_classes:
      nok = 0
    elif len(weights_list) != num_classes:
      print(weights_list)
      print(len(weights_list))
      print(num_classes)
      print("Number of weights and number of classes did not match.")
      print()

  weights_list = [float(i)/sum(weights_list) for i in weights_list]

  nok = 1
  y_minority = 0
  while nok:
    y_minority = int(input("Target class (0 or 1)? "))        # over sampling percentage
    if y_minority == 0 or y_minority == 1:
      nok = 0
    else:
      print("Please enter valid class label.")
      print()

  nok = 1
  os_percent = 200
  while nok:
    os_percent = int(input("Over-sampling % (should be multiple of 100)? "))        # over sampling percentage
    if(os_percent % 100 == 0):
      nok = 0
    else:
      print("Please enter a multiple of 100.")
      print()

  k = int(input("k (number of nearest neighbours)? "))                              # for selecting k nearest neighbors


  X,y = make_classification(n_samples=num_samples,n_features=num_features,n_classes=num_classes,weights=weights_list,random_state=0)
  old_X = X
  old_y = y
  print(len(old_X[:][y==y_minority]))
  print(len(old_X[:][y!=y_minority]))
  print()

  plt.scatter(X[:,0][y==y_minority],X[:,1][y==y_minority],c='g',marker='X')
  plt.scatter(X[:,0][y!=y_minority],X[:,1][y!=y_minority],c='r',marker='.')
  plt.show()


  # Applying SMOTE 

  nnarray = get_knn(X, y, y_minority, k)
  smote_X = []
  smote_y = []
  new_X, new_y = nn_vect_select(X, y, y_minority, nnarray, k, os_percent)

  plt.scatter(new_X[:,0][new_y==y_minority],new_X[:,1][new_y==y_minority],c='g',marker='X')
  plt.scatter(new_X[:,0][new_y!=y_minority],new_X[:,1][new_y!=y_minority],c='r',marker='.')
  plt.show()

  smote_X = new_X
  smote_y = new_y
  print(len(smote_X[:][new_y==y_minority]))
  print(len(smote_X[:][new_y!=y_minority]))
  print()

  X_train,X_test,y_train,y_test = train_test_split(old_X,old_y,test_size=0.2,random_state=0)
  smX_train,smX_test,smy_train,smy_test = train_test_split(smote_X,smote_y,test_size=0.2,random_state=0)

  model_old = KNeighborsClassifier()
  model_old.fit(X_train,y_train)
  y_predict = model_old.predict(X_test)
  print(accuracy_score(y_test,y_predict))
  print(pd.crosstab(y_test,y_predict))
  print('False Negatives : ',len(y_test[(y_predict!=y_minority) & (y_test==y_minority)]))
  print('True Positives : ',len(y_predict[(y_predict==y_minority) & (y_test==y_minority)]))
  print('False Positives : ',len(y_predict[(y_predict==y_minority) & (y_test!=y_minority)]))
  print('True Negatives : ',len(y_predict[(y_predict!=y_minority) & (y_test!=y_minority)]))
  print('Accuracy of Class 0',(100*len(y_test[(y_predict==y_minority) & (y_test==y_minority)])/(len(y_test[(y_predict!=y_minority) & (y_test==y_minority)])+len(y_predict[(y_predict==y_minority) & (y_test==y_minority)]))),'%')
  print('Accuracy of Class 1',(100*len(y_test[(y_predict!=y_minority) & (y_test!=y_minority)])/(len(y_test[(y_predict==y_minority) & (y_test!=y_minority)])+len(y_predict[(y_predict!=y_minority) & (y_test!=y_minority)]))),'%')
  print('Model Apparent Accuracy : ',100*accuracy_score(y_test,y_predict),'%')

  print()
  print('After SMOTE')
  print()
  model_new = KNeighborsClassifier()
  model_new.fit(smX_train,smy_train)
  smy_predict = model_new.predict(smX_test)
  print(accuracy_score(smy_test,smy_predict))
  print(pd.crosstab(smy_test,smy_predict))
  print('False Negatives : ',len(smy_test[(smy_predict!=y_minority) & (smy_test==y_minority)]))
  print('True Positives : ',len(smy_predict[(smy_predict==y_minority) & (smy_test==y_minority)]))
  print('False Positives : ',len(smy_predict[(smy_predict==y_minority) & (smy_test!=y_minority)]))
  print('True Negatives : ',len(smy_predict[(smy_predict!=y_minority) & (smy_test!=y_minority)]))
  print('Accuracy of Class 0',(100*len(smy_test[(smy_predict==y_minority) & (smy_test==y_minority)])/(len(smy_test[(smy_predict!=y_minority) & (smy_test==y_minority)])+len(smy_predict[(smy_predict==y_minority) & (smy_test==y_minority)]))),'%')
  print('Accuracy of Class 1',(100*len(smy_test[(smy_predict!=y_minority) & (smy_test!=y_minority)])/(len(smy_test[(smy_predict==y_minority) & (smy_test!=y_minority)])+len(smy_predict[(smy_predict!=y_minority) & (smy_test!=y_minority)]))),'%')
  print('Model Apparent Accuracy : ',100*accuracy_score(smy_test,smy_predict),'%')





# Another way of over-sampling
# X,y = make_classification(n_samples=1000,n_features=4,n_classes=2,weights=[0.01,0.99],random_state=0)
# old_X = X
# old_y = y
# print(X.shape,y.shape)
# print(type(X),type(y))

# plt.scatter(X[:,0][y==0],X[:,1][y==0],c='g',marker='X')
# plt.scatter(X[:,0][y==1],X[:,1][y==1],c='r',marker='.')
# plt.show()

# '''
#   Random vector selection
# '''

# max_size = len(X[:][y==0])
# target_size = len(X[:][y==1])
# comp = 0
# diff = max_size
# while max_size < target_size:
#   if comp != int(100*(max_size - diff)/(target_size - diff)):
#     print(int(100*(max_size - diff)/(target_size - diff)),'% Completed')
#     comp = int(100*(max_size - diff)/(target_size - diff))
#   max_size = len(X[:][y==0])
#   index_1 = random.randint(0,max_size-1)
#   index_2 = random.randint(0,max_size-1)
#   ratio = random.random()
#   new_index = X[:][y==0][index_1] + ratio*(X[:][y==0][index_2] - X[:][y==0][index_1])
#   X = np.append(X,[new_index],axis=0)
#   y = np.append(y,[0])

# '''
#   NN vector selection
# '''

# max_size = len(X[:][y==0])
# target_size = len(X[:][y==1])/2
# comp = 0
# diff = max_size
# while max_size < target_size:
#   max_size = len(X[:][y==0])
#   index_1 = random.randint(0,max_size-1)
#   index_2 = 0 
#   c_index = 0
#   min_dist = -1
#   while c_index < len(X[:][y==0]):
#     # print(len(X[:][y==0]))
#     if c_index != index_1 and min_dist == -1 or np.sum(np.square(X[:][y==0][index_1]-X[:][y==0][c_index])) < c_dist:
#       c_dist = np.sum(np.square(X[:][y==0][index_1]-X[:][y==0][c_index]))
#       index_2 = c_index
#     c_index += 1
    
#   ratio = random.random()
#   new_index = X[:][y==0][index_1] + ratio*(X[:][y==0][index_2] - X[:][y==0][index_1])
#   X = np.append(X,[new_index],axis=0)
#   y = np.append(y,[0])

# plt.scatter(X[:,0][y==0],X[:,1][y==0],c='g',marker='X')
# plt.scatter(X[:,0][y==1],X[:,1][y==1],c='r',marker='.')
# plt.show()


