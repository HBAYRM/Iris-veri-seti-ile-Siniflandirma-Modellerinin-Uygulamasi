# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 16:18:21 2022

@author: hbayr
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme------
#--------------------------------------------------
#2.1.veri yukleme
veriler = pd.read_csv('iris.csv')
print(veriler)
x=veriler.iloc[:,:4].values
y=veriler.iloc[:,4:].values

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
#KArsılastırma matriksi ile basarı yuzdesı gormme
from sklearn.metrics import confusion_matrix

#Lojıstık Regresyon
from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=(0))
logr.fit(X_train, y_train)
y_pred=logr.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print("Logistic R.")
print(cm)

#KNN ALGORİTMASI
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train, y_train)
y_predknn=knn.predict(X_test)
cm2=confusion_matrix(y_test,y_predknn)
print("KNN")
print(cm2)

#SVM(Support VEktor MAchine)
from sklearn.svm import SVC
svc=SVC(kernel="linear")
svc.fit(X_train, y_train)
y_predsvm=svc.predict(X_test)
cm3=confusion_matrix(y_test,y_predsvm)
print("SVM")
print(cm3)

#Naive Bayes Yontemi
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)
y_prednaivebayer=nb.predict(X_test)
cm4=confusion_matrix(y_test,y_prednaivebayer)
print("Naive Bayes")
print(cm4)

#Decision Tree Yöntemi
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, y_train)
y_preddt=dtc.predict(X_test)
cm5=confusion_matrix(y_test,y_preddt)
print("Decision Tree")
print(cm5)

#Random Forest(Rassal Orman)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion="entropy")
rfc.fit(X_train,y_train)
y_predrfc=rfc.predict(X_test)
cm6=confusion_matrix(y_test,y_predrfc)
print("Random Forest")
print(cm6)

#ROC Hesaplama(FPR,TPR)
from sklearn import metrics
y_proba=rfc.predict_proba(X_test)
fpr,tpr,thold=metrics.roc_curve(y_test,y_proba[:,0],pos_label="e")
print(fpr)
print(tpr)


