import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors, naive_bayes, metrics


class Dataset():
    def __init__(self, attributeList:list, trainDataList:np.ndarray, testDataList:np.ndarray):
        self.attributeList = attributeList
        self.trainDataList = trainDataList
        self.testDataList = testDataList
    
    def convertToScikitDataset(self):
        self.pandasData = pd.DataFrame(np.concatenate([self.trainDataList, self.trainDataList]), columns=[attr.name for attr in self.attributeList])
        # print(self.pandasData.shape)
        # print(self.pandasData.keys)
        _scikitDataset = Bunch()
        _scikitDataset['target'] = self.pandasData['class']
        _scikitDataset['data'] = self.pandasData.loc[:, :self.pandasData.columns.values[-2]]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(_scikitDataset['data'], _scikitDataset['target'], random_state=0)
        # print("X_train shape:", self.X_train.shape)
        # print("X_test shape:", self.X_test.shape)

        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def scikitLinearSVC(self, X_train, X_test, Y_train, Y_test):
        print("--- LinearSVC ---")
        _LSVC = svm.LinearSVC(max_iter=10000)
        _LSVC.fit(X_train, Y_train)
        _accuracy = _LSVC.score(X_test, Y_test)
        print("正答率", _accuracy)

        print("詳細結果")
        print(metrics.classification_report(Y_test, _LSVC.predict(X_test)))
    
    def scikitKNeighborsClassifier(self, X_train, X_test, Y_train, Y_test):
        print("--- KNeighborsClassifier ---")
        _KNC = neighbors.KNeighborsClassifier()
        _KNC.fit(X_train, Y_train)
        _accuracy = _KNC.score(X_test, Y_test)
        print("正答率", _accuracy)

        print("詳細結果")
        print(metrics.classification_report(Y_test, _KNC.predict(X_test)))
    
    def scikitSVC(self, X_train, X_test, Y_train, Y_test):
        print("--- SVC ---")
        _SVC = svm.SVC(max_iter=10000)
        _SVC.fit(X_train, Y_train)
        _accuracy = _SVC.score(X_test, Y_test)
        print("正答率", _accuracy)

        print("詳細結果")
        print(metrics.classification_report(Y_test, _SVC.predict(X_test)))
    
    def scikitNaiveBayes(self, X_train, X_test, Y_train, Y_test):
        print("--- Naive Bayes ---")
        _GNB = naive_bayes.GaussianNB()
        _GNB.fit(X_train, Y_train)
        _accuracy = _GNB.score(X_test, Y_test)
        print("正答率", _accuracy)

        print("詳細結果")
        print(metrics.classification_report(Y_test, _GNB.predict(X_test)))