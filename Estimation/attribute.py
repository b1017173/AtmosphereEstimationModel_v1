class Attribute:
    def __init__(self, attrName:str, attrType:str):
        self.name = attrName
        self.type = attrType
    
    def setClassValues(self, values):
        self.classValues = values