class MulticlassLabelAssigner():
    def __init__(self, orderList:list = None):
        self.txtLabelDict = dict()
        self.labelTxtDict = dict()
        # this will increase the label one by one and later on return the dictionary
        self.maxIndex = -1
        self.orderList = orderList


    def assign(self, inputTxt:str)->int:

        if self.orderList is not None:
            return self.orderList.index(inputTxt)

        if inputTxt in self.txtLabelDict:
            return self.txtLabelDict.get(inputTxt)
        else:
            self.maxIndex += 1
            self.txtLabelDict[inputTxt] = self.maxIndex
            self.labelTxtDict[self.maxIndex] = inputTxt
            return self.txtLabelDict.get(inputTxt)

    def getTxtLabelDict(self)->dict:
        return self.txtLabelDict

    def getLabelTxtDict(self)->dict:
        return self.labelTxtDict
