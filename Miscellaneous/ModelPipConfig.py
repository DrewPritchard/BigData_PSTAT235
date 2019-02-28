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
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.feature import OneHotEncoder
from ModelEvaluators.MultiClassLogLossEvaluator import MultiClassLogLossEvaluator

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

        outlierSmoother = OutlierSmoother(thresh=5.5,
                                          inputCols=["latitude", "longitude", "out_price"],
                                          outputCols=["out_latitude", "out_longitude", "out_price2"])

        assemblerForGMM = VectorAssembler(inputCols=["out_latitude", "out_longitude"],
                                          outputCol="gmmFeatures")


        gmm = GaussianMixture(featuresCol="gmmFeatures",
                              predictionCol="gmmPrediction",
                              k=7,
                              probabilityCol="gmmAssignmentProbability",
                              tol=0.01,
                              maxIter=100,
                              seed=None)

        gmmLabelOneHotEncoder = OneHotEncoder(inputCol="gmmPrediction", outputCol="gmmPredictionVector")

        assembler = VectorAssembler(inputCols=["out_bathrooms",
                                               "out_bedrooms",
                                               "out_created",
                                               "out_price2",
                                               "gmmPredictionVector"],
                                    outputCol="features")

        # self.modelEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
        self.modelEvaluator = MultiClassLogLossEvaluator(probabilityVectorCol="probability", labelCol="label")

        if self.method == "LogisticRegression":
            Logger.logger.info("Using the logistic regression")
            logisticR = LogisticRegression(maxIter=20, family="multinomial")  # , regParam=0.3, elasticNetParam=0.8)

            self.paramGrid = ParamGridBuilder().addGrid(logisticR.regParam,
                                                   [0.1, 0.01]).addGrid(logisticR.elasticNetParam,
                                                                        [0, 1]).build()
            self.estimator = logisticR

        if self.method == "RandomForest":
            Logger.logger.info("Using the RandomForest")
            rf = RandomForestClassifier(numTrees=10)
            self.paramGrid = ParamGridBuilder().addGrid(gmm.k, [2, 10]).build()
            self.estimator = rf

        self.stages = [imputer,
                       outlierSmoother,
                       assemblerForGMM,
                       gmm,
                       gmmLabelOneHotEncoder,
                       assembler,
                       self.estimator]

    def getStages(self):
        return self.stages

    def getParamGrid(self):
        return self.paramGrid

    def getEstimator(self):
        return self.estimator

    def getModelEvaluator(self):
        return self.modelEvaluator
