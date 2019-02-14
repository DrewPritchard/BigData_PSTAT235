from pyspark.ml.feature import Imputer
from pyspark.ml.classification import LogisticRegression, OneVsRest, RandomForestClassifier
from FeaturesMakers import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.feature import StandardScaler
from Miscellaneous.Logger import Logger

class PipConfig(object):

    def __init__(self):
        return

    def getStages(self, method="LogisticRegression"):

        imputer = Imputer(strategy="mean",
                          inputCols=["bathrooms", "bedrooms", "created", "price"],
                          outputCols=["out_bathrooms", "out_bedrooms", "out_created", "out_price"])

        assembler = VectorAssembler(inputCols=["out_bathrooms", "out_bedrooms", "out_created", "out_price"],
                                    outputCol="features")


        modelEvaluator = MulticlassClassificationEvaluator()

        estimator = None
        paramGrid = None

        if method == "LogisticRegression":
            Logger.logger.info("Using the logistic regression")
            logisticR = LogisticRegression(maxIter=20, family="multinomial")  # , regParam=0.3, elasticNetParam=0.8)

            paramGrid = ParamGridBuilder().addGrid(logisticR.regParam,
                                                   [0.1, 0.01]).addGrid(logisticR.elasticNetParam,
                                                                        [0, 1]).build()
            estimator = logisticR


        if method == "RandomForest":
            Logger.logger.info("Using the RandomForest")
            rf = RandomForestClassifier()
            paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [3,10]).build()
            estimator = rf


        crossval = CrossValidator(estimator=estimator,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=modelEvaluator,
                                  numFolds=5)
        stages = [imputer, assembler, crossval]
        return stages

