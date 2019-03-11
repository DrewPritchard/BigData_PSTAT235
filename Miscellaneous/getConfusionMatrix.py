import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def getConfusionMatrix(inFile: str, weight):

    probabilities = list(map(eval,pd.read_csv(inFile)["probability"].tolist()))
    labels = pd.read_csv(inFile)["label"].tolist()

    def getPredictionLabel(probabilities, weight):
        # Make sure the indicies correspond with labels
        output = []
        for currProbVec in probabilities:
            output.append(np.argmax( np.array(currProbVec)*np.array(weight) ))
        return output

    return confusion_matrix(labels,getPredictionLabel(probabilities, weight))


print(getConfusionMatrix("../run4report/predictionOutput_RF_tune_depth.csv", [5,1,1]))


