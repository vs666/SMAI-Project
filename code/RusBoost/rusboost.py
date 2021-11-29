# -*- coding: utf-8 -*-
"""RusBoost.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MQ6LcYF4g-C64Mgsdf81l7rs7X2uDOAy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import sys 
import random
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

class RusBoost:
  def __init__(self,n_classifiers,base_classifier,percentage_oversample,learning_rate,supports_class_probabilities=False):
    self.n_classifiers = n_classifiers
    self.algorithm='SAMME'
    if supports_class_probabilities:
      self.algorithm='SAMME.R'
    
    self.classifier = AdaBoostClassifier(base_estimator=base_classifier,n_estimators=self.n_classifiers,learning_rate=learning_rate)
    self.percentage_oversample = percentage_oversample

  def fit(self,X,y,majority_class,minority_class):
    x_final,y_final = self.Rus(X,y,majority_class,minority_class)
    self.classifier.fit(x_final,y_final)
    return self.classifier.score(x_final,y_final)

  def predict(self,x_predict):
    return self.classifier.predict(x_predict)
  
  def Rus(self,x_in, y_in, oversample_label=1, undersample_label=0 ):
    assert(self.percentage_oversample > 0)
    x_oversampled = x_in[:][y_in==oversample_label]
    x_undersampled = x_in[:][y_in==undersample_label]
    assert(len(x_oversampled) >= len(x_undersampled))
    selected_list = np.random.choice(len(x_oversampled),int(len(x_undersampled)*self.percentage_oversample/(100-self.percentage_oversample)),replace=False)
    x_selected = []
    y_final = []
    for i in selected_list:
      x_selected.append(x_oversampled[i])
      y_final.append(oversample_label)
    for i in x_undersampled:
      x_selected.append(i)
      y_final.append(undersample_label)
    x_selected = np.array(x_selected)#.reshape(len(selected_list),x_oversampled.shape[1])
    y_final = np.array(y_final).reshape(len(y_final),)
    shuffler = np.random.permutation(len(y_final))
    x_selected = x_selected[shuffler]
    y_final = y_final[shuffler]
    return x_selected,y_final



def makeData(minority_fraction, n_features,n_samples,test_size=0.2):
  X,y = make_classification(n_samples=n_samples,n_features=n_features,n_classes=2,weights=[minority_fraction,1-minority_fraction],random_state=0)
  X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = test_size, random_state = 1)
  return X_train,X_test,y_train,y_test

def getClassifier(classifier_label):
  classifier = DecisionTreeClassifier() # by default
  if classifier_label == 'SVM':
    classifier = SVC(kernel='rbf',probability=True)
  elif classifier_label == 'DecisionTree':
    classifier = DecisionTreeClassifier()
  return classifier

def main(classifier_label,minority_fraction,n_features,n_samples,n_classifiers,test_size=0.2):
  classifier = getClassifier(classifier_label)
  X_train,X_test,y_train,y_test = makeData(minority_fraction,n_features,n_samples,test_size)
  
  model = RusBoost(n_classifiers,classifier,60,0.5)
  score = model.fit(X_train,y_train,1,0)
  print(score)
  y_pred = model.predict(X_test)
  mino = [0,0]
  majo = [0,0]
  for i in range(len(y_test)):
    if y_test[i] == 1 and y_pred[i] == 0:
      majo[0]+=1
    elif y_test[i] == 1 and y_pred[i] == 1:
      majo[1] += 1
    elif y_test[i] == 0 and y_pred[i] == 0:
      mino[0]+=1
    elif y_test[i] == 0 and y_pred[i] == 1:
      mino[1] += 1
  print('Majority Accuracy ',majo[1]/sum(majo)*100)
  print('Minority Accuracy',mino[0]/sum(mino)*100)
  print('\t\ty-pred=0\ty-pred=1')
  print('y-true = 1\t',majo[0],'|\t',majo[1])
  print('y-true = 0\t',mino[0],'|\t',mino[1])


if __name__ == '__main__':
  pass 
  # Write code here to call main()