from pyspark.ml.feature import Imputer
from pyspark.ml.classification import LogisticRegression, OneVsRest
from FeaturesMakers import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder

from pyspark.ml.feature import StandardScaler

class PipConfig(object):

    def __init__(self):

        imputer = Imputer(strategy="mean",
                          inputCols=["bathrooms", "bedrooms", "created", "price"],
                          outputCols=["out_bathrooms", "out_bedrooms", "out_created", "out_price"])

        assembler = VectorAssembler(inputCols=["out_bathrooms", "out_bedrooms", "out_created","out_price"], outputCol="features")

        logisticR = LogisticRegression(maxIter=20, family="multinomial")  # , regParam=0.3, elasticNetParam=0.8)

        self.stages = [imputer, assembler, logisticR]

        self.paramGrid = ParamGridBuilder().addGrid(logisticR.regParam,
                                               [0.1, 0.01]).addGrid(logisticR.elasticNetParam,
                                                                    [0, 1]).build()


    def getStages(self):
        return self.stages

    def getParamGrid(self):
        return self.paramGrid