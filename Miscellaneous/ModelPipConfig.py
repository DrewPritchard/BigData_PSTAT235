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
from FeaturesMakers.OutlierSmoother import OutlierSmoother

class PipConfig(object):

    def __init__(self, method=None):
        self.estimator = None
        self.paramGrid = None
        if method is None:
            self.method = "LogisticRegression"
        else:
            self.method = method
        self.stages = None

        imputer = Imputer(strategy="mean",
                          inputCols=["bathrooms", "bedrooms", "created", "price"],
                          outputCols=["out_bathrooms", "out_bedrooms", "out_created", "out_price"])

        outlierSmoother = OutlierSmoother(thresh=3.5,
                                          inputCols=["latitude", "longitude"],
                                          outputCols=["out_latitude", "out_longitude"])


        assembler = VectorAssembler(inputCols=["out_bathrooms",
                                               "out_bedrooms",
                                               "out_created",
                                               "out_price",
                                               "out_latitude",
                                               "out_longitude"],
                                    outputCol="features")

        self.modelEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

        if self.method == "LogisticRegression":
            Logger.logger.info("Using the logistic regression")
            logisticR = LogisticRegression(maxIter=20, family="multinomial")  # , regParam=0.3, elasticNetParam=0.8)

            self.paramGrid = ParamGridBuilder().addGrid(logisticR.regParam,
                                                   [0.1, 0.01]).addGrid(logisticR.elasticNetParam,
                                                                        [0, 1]).build()
            self.estimator = logisticR

        if self.method == "RandomForest":
            Logger.logger.info("Using the RandomForest")
            rf = RandomForestClassifier()
            self.paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [3,10]).build()
            self.estimator = rf

        self.stages = [imputer, outlierSmoother, assembler, self.estimator]

    def getStages(self):
        return self.stages

    def getParamGrid(self):
        return self.paramGrid

    def getEstimator(self):
        return self.estimator

    def getModelEvaluator(self):
        return self.modelEvaluator
