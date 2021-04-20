from FeatureExtraction.src.dataset import Dataset
from FeatureExtraction.src.attribute import Attribute
from FeatureExtraction.src.speechState import SpeechState

import sys
import numpy as np
import pandas as pd
import openpyxl as op

def readSTExcel():
    speak_time_excel_path = sys.argv[1]
    futures_attr_path = sys.argv[2]

    wb = pd.ExcelFile(speak_time_excel_path, engine='openpyxl') # ワークブックの読み込み
    shs = []
    for sh_name in wb.sheet_names:                              # 全シートループ
        sh = wb.parse(sh_name, usecols='B:IG')                  # シートの読み込み
        shs.append(SpeechState(sh))
    
    return shs

def readArff():
    # train_path = sys.argv[1]
    # test_path = sys.argv[2]
    train_path = 'data/segment_train.arff'
    test_path = 'data/segment_test.arff'
    attributeList, trainDataList = readTrainArff(train_path)
    testDataList = readTestArff(test_path)
    dataset = Dataset(attributeList, trainDataList, testDataList)

def readTrainArff(train_path:str):
    attributeList = []
    dataList:np.ndarray = [[]]
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
            dataList = np.append(dataList, line.split(','))
    return attributeList, dataList

def readTestArff(test_path:str):
    dataList:np.ndarray = [[]]
    test_file = open(test_path)
    lines = [line.rstrip('\n') for line in test_file]
    for i, line in enumerate(lines):
        if(not line.startswith('@data') and not line.startswith('@relation') and not line.startswith('%')):
            dataList = np.append(dataList, line.split(','))
    return dataList

def main():
    # shs = readSTExcel()
    readArff()

if __name__ == '__main__':
    main()