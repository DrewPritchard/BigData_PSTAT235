from pyspark.ml.evaluation import Evaluator
import numpy as np
from Miscellaneous.Logger import Logger
class MultiClassLogLossEvaluator(Evaluator):
    def __init__(self, probabilityVectorCol="probability", labelCol="label"):
        self.probabilityVectorCol = probabilityVectorCol
        self.labelCol = labelCol

    @staticmethod
    def computeLogLoss(labels, probVecs):
        if len(labels) != len(probVecs):
            raise ValueError("the length of the labels must be equal to the length of the probVecs")
        if min(labels) < 0 or max(labels) > len(probVecs):
            raise ValueError("Please ensure the labels are 0 based")
        N = len(labels)
        M = len(probVecs[0])
        result = 0.0
        for i in range(N):
            for j in range(M):
                currVec = probVecs[i]
                result = result + (int(labels[i]) == int(j))*np.log(currVec[j])
        return -1*result/(N*1.0)

    def evaluate(self, dataset):
        normalize = lambda vec: vec / (vec).sum()
        probVecs = [normalize(np.vectorize(lambda p: max(min(p, 1 - 10 ** (-7)), 10 ** (-7)))((row[self.probabilityVectorCol]).toArray())) for row in dataset.collect()]
        labels = [int(row[self.labelCol]) for row in dataset.collect()]
        result = MultiClassLogLossEvaluator.computeLogLoss(labels, probVecs)
        Logger.logger.info("the logloss is: " + str(result))
        return result

    def isLargerBetter(self):
        return False