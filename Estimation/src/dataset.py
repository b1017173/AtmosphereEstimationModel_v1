import numpy as np
from numpy.random import seed
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm, neighbors, naive_bayes, metrics

class Dataset():
    # プロパティ
    # attributeList : 特徴量の名前リスト
    # trainDataList : 二次元データ
    # pandasData    : attributeListとtrainDataListを組み合わせた全データ
    # selectedData  : 選択データ
    def __init__(self, attributeList:list, trainDataList:np.ndarray):
        self.attributeList = attributeList
        self.trainDataList = trainDataList
        self.pandasData = pd.DataFrame(self.trainDataList, columns=[attr.name for attr in self.attributeList])
        self.reformatPandasData()

    def machineLearning(self, speakers_num:int):
        # エラーハンドリング
        if speakers_num < 1 or 4 < speakers_num:
            print("Error: speaker_numは1~4で指定")
            return
        self.dataSelection(speakers_num)
        self.preFutureSelection(speakers_num)
        self.convertToScikitDataset()
        self.scikitLearningByCrossVal()
        self.scikitLinearSVC()
        self.scikitKNeighborsClassifier()
        self.scikitLinearSVC()
        self.scikitNaiveBayes()

    def reformatPandasData(self):
        _futures = self.pandasData.drop(columns = 'class')
        _futures = _futures.astype(float)
        _class = self.pandasData[['class']]
        _class = _class.astype(str)
        self.pandasData = pd.concat([_futures, _class], axis = 1)
        # print(self.pandasData.dtypes)
    
    def dataSelection(self, speakers_num:int):
        self.selectedData:DataFrame = None
        for index, data in self.pandasData.iterrows():
            if float(data['f{0}'.format((speakers_num - 1) * 6 + 6)]):
                if self.selectedData is None:
                    # self.selectedData = data
                    self.selectedData = pd.DataFrame([data])
                else:
                    self.selectedData = self.selectedData.append(data, ignore_index=True)
        # print(self.selectedData)

    def preFutureSelection(self, speakers_num:int):
        # 各カテゴリーの人数依存リスト
        _category_valids = [\
            [True, True, False, False, False, True, False, False, False, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False],\
            [True, True, True, False, False, True, True, False, False, True, True, True, True, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, True, False, False, True, True, False, False],\
            [True, True, True, True, False, True, True, True, False, True, True, True, True, True, False, True, True, False, True, False, False, True, True, False, True, False, False, True, True, True, False, True, True, True, False],\
            [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]\
        ]
        _new_selected_data:DataFrame = self.selectedData.loc[:, 'class']
        for i, isValid in enumerate(_category_valids[speakers_num - 1]):
            if isValid:
                try:
                    _selected_column = self.selectedData.iloc[:, i*6:(i*6)+6]
                    if _new_selected_data is None:
                        _new_selected_data = _selected_column
                    else:
                        _new_selected_data = pd.concat([_new_selected_data, _selected_column], axis = 1)
                except:
                    print("ぬるぽ")

        self.selectedData = _new_selected_data
        # print(self.selectedData.dtypes)
    
    def convertToScikitDataset(self):
        # print(self.selectedData.shape)
        # print(self.selectedData.keys)
        _scikitDataset = Bunch()
        _scikitDataset['target'] = self.selectedData['class']
        _scikitDataset['data'] = self.selectedData.drop(columns = 'class')
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(_scikitDataset['data'], _scikitDataset['target'], random_state=0)
        print("X_train shape:", self.X_train.shape)
        print("X_test shape:", self.X_test.shape)

    def scikitLinearSVC(self):
        print("--- LinearSVC ---")
        _LSVC = svm.LinearSVC(max_iter=1000000)
        _LSVC.fit(self.X_train, self.Y_train)
        _accuracy = _LSVC.score(self.X_test, self.Y_test)
        print("正答率", _accuracy)

        print("詳細結果")
        print(metrics.classification_report(self.Y_test, _LSVC.predict(self.X_test)))
    
    def scikitKNeighborsClassifier(self):
        print("--- KNeighborsClassifier ---")
        _KNC = neighbors.KNeighborsClassifier()
        _KNC.fit(self.X_train, self.Y_train)
        _accuracy = _KNC.score(self.X_test, self.Y_test)
        print("正答率", _accuracy)

        print("詳細結果")
        print(metrics.classification_report(self.Y_test, _KNC.predict(self.X_test)))
    
    def scikitSVC(self):
        print("--- SVC ---")
        _SVC = svm.SVC(max_iter=10000)
        _SVC.fit(self.X_train, self.Y_train)
        _accuracy = _SVC.score(self.X_test, self.Y_test)
        print("正答率", _accuracy)

        print("詳細結果")
        print(metrics.classification_report(self.Y_test, _SVC.predict(self.X_test)))
    
    def scikitNaiveBayes(self):
        print("--- Naive Bayes ---")
        _GNB = naive_bayes.GaussianNB()
        _GNB.fit(self.X_train, self.Y_train)
        _accuracy = _GNB.score(self.X_test, self.Y_test)
        print("正答率", _accuracy)

        print("詳細結果")
        print(metrics.classification_report(self.Y_test, _GNB.predict(self.X_test)))

    def scikitLearningByCrossVal(self, k:int = 5):
        _scikitDataset = Bunch()
        _scikitDataset['target'] = self.selectedData['class']
        _scikitDataset['data'] = self.selectedData.drop(columns = 'class')

        _scores = cross_val_score(svm.LinearSVC(max_iter=1000000), _scikitDataset['data'], _scikitDataset['target'], cv = k)
        print('LinearSVC Cross-Validation scores: ', _scores)
        print('Average score: ', np.mean(_scores))
        print('\n')

        _scores = cross_val_score(neighbors.KNeighborsClassifier(), _scikitDataset['data'], _scikitDataset['target'], cv = k)
        print('KNeighborsClassifier Cross-Validation scores: ', _scores)
        print('Average score: ', np.mean(_scores))
        print('\n')

        _scores = cross_val_score(svm.SVC(max_iter=10000), _scikitDataset['data'], _scikitDataset['target'], cv = k)
        print('SVC Cross-Validation scores: ', _scores)
        print('Average score: ', np.mean(_scores))
        print('\n')

        _scores = cross_val_score(naive_bayes.GaussianNB(), _scikitDataset['data'], _scikitDataset['target'], cv = k)
        print('NaiveBayes Cross-Validation scores: ', _scores)
        print('Average score: ', np.mean(_scores))
        print('\n')