from attribute import Attribute
from dataset import Dataset
import numpy as np
import pandas as pd

def machineLearning(dataset):
    X_train, X_test, Y_train, Y_test = dataset.convertToScikitDataset()
    dataset.scikitLinearSVC(X_train, X_test, Y_train, Y_test)
    dataset.scikitKNeighborsClassifier(X_train, X_test, Y_train, Y_test)
    dataset.scikitSVC(X_train, X_test, Y_train, Y_test)
    dataset.scikitNaiveBayes(X_train, X_test, Y_train, Y_test)

def readArff():
    # train_path = sys.argv[1]
    # train_path = 'data/segment_train.arff'
    train_path = 'data/excitement_train.arff'
    # train_path = 'data/estimation_train.arff'
    attributeList, trainDataList = readTrainArff(train_path)
    # print('attributeList:', attributeList)
    # print('trainDataList:', trainDataList)
    dataset = Dataset(attributeList, trainDataList)
    return dataset

def readTrainArff(train_path:str):
    attributeList = []
    dataList:np.ndarray = []
    train_file = open(train_path)
    lines = [line.rstrip('\n') for line in train_file]
    for i, line in enumerate(lines):
        if(line.startswith('@attribute')):
            splited_lines = line.split(' ', 2)
            if splited_lines[2].startswith('numeric') or splited_lines[2].startswith('NUMERIC'):
                attr = Attribute(splited_lines[1], 'numeric')
                attributeList.append(attr)
            elif splited_lines[2].startswith('{'):
                attr = Attribute(splited_lines[1], 'class')
                splited_lines[2]
                class_values = splited_lines[2].translate(str.maketrans({'{':None, '}':None}))
                class_values = [x.strip(" ") for x in class_values.split(",")]
                attr.setClassValues(class_values)
                attributeList.append(attr)
            else:
                print('未対応の特徴量形式が書き込まれています．')
        elif(not line.startswith('@data') and not line.startswith('@relation') and not line.startswith('%')):
            if(np.size(dataList) == 0):
                dataList = [line.split(',')]
            else:
                dataList = np.insert(dataList, 0, line.split(','), axis=0)
    dataList = np.where(dataList == '?', np.NaN, dataList)                              # 欠損値をnumpy用に置換
    row_contain_nan = ~np.isnan(np.delete(dataList, -1, 1).astype(float)).any(axis=1)   # 欠損値のある行の検知
    dataList = dataList[row_contain_nan]                                                # 欠損値のある行を消去
    return attributeList, dataList

def main():
    dataset = readArff()
    machineLearning(dataset)

if __name__ == '__main__':
    main()
