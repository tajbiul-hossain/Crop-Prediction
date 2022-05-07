from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')


def crop_pred(inputData):
    data = pd.read_csv("Crop_recommendation.csv")

    features = data[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    target = data['label']
    labels = data['label']

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


    DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)
    DecisionTree.fit(Xtrain,Ytrain)

    NaiveBayes = GaussianNB()
    NaiveBayes.fit(Xtrain,Ytrain)

    LogReg = LogisticRegression(random_state=2)
    LogReg.fit(Xtrain,Ytrain)

    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain,Ytrain)

    vc = VotingClassifier([('DT', DecisionTree), ('LogReg', LogReg), ('RF', RF), ('NB', NaiveBayes)], voting='soft')
    vc.fit(Xtrain,Ytrain)
    
    inputData = np.array(inputData)
    inputData = inputData.reshape((1,-1))

    return vc.predict(inputData)
