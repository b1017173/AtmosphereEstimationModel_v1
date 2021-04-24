import numpy as np
import pandas as pd

class Dataset():
    def __init__(self, attributeList:list, trainDataList:np.ndarray, testDataList:np.ndarray):
        self.attributeList = attributeList
        self.trainDataList = trainDataList
        self.testDataList = testDataList
    
    def convertToScikitDataset(self):
        self.pandasData = pd.DataFrame(np.concatenate([self.trainDataList, self.trainDataList]), columns=[attr.name for attr in self.attributeList])

        print(self.pandasData)
        return 