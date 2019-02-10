class MulticlassLabelAssigner():
    def __init__(self):
        self.labelDict = dict()
        # this will increase the label one by one and later on return the dictionary
        self.maxIndex = -1
    def assign(self, inputTxt:str)->int:
        if inputTxt in self.labelDict:
            return self.labelDict.get(inputTxt)
        else:
            self.maxIndex += 1
            self.labelDict[inputTxt] = self.maxIndex
            return self.labelDict.get(inputTxt)

    def getDict(self)->dict:
        return self.labelDict