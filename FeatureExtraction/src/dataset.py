import numpy as np

class Dataset():
    def __init__(self, attributeList:list, trainDataList:np.ndarray, testDataList:np.ndarray):
        self.attributeList = attributeList
        self.trainDataList = trainDataList
        self.testDataList = testDataList