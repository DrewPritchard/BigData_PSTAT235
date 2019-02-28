import sys
import re
import pandas as pd
from Miscellaneous.Logger import Logger


from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql.types import *       # for datatype conversion
from pyspark.sql.functions import *   # for col() function
from pyspark.ml.linalg import DenseVector,VectorUDT,SparseVector
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression
import pandas as pd
import numpy as np
from pyspark.ml.classification import LogisticRegression
from sklearn.metrics import confusion_matrix

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from FeaturesMakers.MulticlassLabelAssigner import MulticlassLabelAssigner

from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructField,StructType,DoubleType,ArrayType
import time
import datetime


argsArr = sys.argv[1:]
argsDict = dict()

# this part of code will enable you to specify the running parameters into the dictionary
# the input format is shown as follows:
# python3 toyDataModelRunner.py re:2 uq:* we:we
# if you specify anything to be * that is None

for inputToken in argsArr:
    key, val = inputToken.split(":")

    if "true" == val.lower():
        argsDict[key] = True
        continue
    if "false" == val.lower():
        argsDict[key] = False
        continue
    if re.match("^\d+?\.\d+?$", val):
        argsDict[key] = float(val)
        continue
    if re.match("[0-9]", val):
        argsDict[key] = int(val)
        continue
    if re.match("(.*)[a-zA-Z]", val):
        argsDict[key] = str(val)
        continue
    argsDict[key] = None

Logger.logger.info(argsDict)

doTest = argsDict.get("doTest", True)

finalClassifier = argsDict.get("finalClassifier", "LogisticRegression")


spark = SparkSession \
    .builder \
    .master("local") \
    .appName("pstat235final") \
    .config("spark.executor.memory", '4g') \
    .config('spark.executor.cores', '1') \
    .config('spark.cores.max', '1') \
    .config("spark.driver.memory",'1g') \
    .getOrCreate()

sc = SparkContext.getOrCreate()
sc.setLogLevel("ERROR")

sqlCtx = SQLContext(sc)

from Miscellaneous.ModelPipConfig import PipConfig


if doTest:
    Logger.logger.info("Testing the model performance")
    input_data_pd = pd.read_json("data/train.json")
    input_data_pd["bedrooms"]+=0.0
    input_data_pd["created"] = input_data_pd["created"].apply(lambda s: time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()))
    input_data_pd["price"] += 0.0
    multiclassLabelAssigner = MulticlassLabelAssigner(["high", "medium", "low"])
    input_data_pd["interest_level"] = input_data_pd["interest_level"].apply(lambda s: multiclassLabelAssigner.assign(inputTxt=s))

    # input_data_pd[["bedrooms"]] = input_data_pd[["bedrooms"]].apply(pd.to_numeric(downcast='float'))
    schema = StructType([StructField('bathrooms', DoubleType(), True),
                         StructField('bedrooms', DoubleType(), True),
                         StructField('building_id', StringType(), True),
                         StructField('created', DoubleType(), True),
                         StructField('description', StringType(), True),
                         StructField('display_address', StringType(), True),
                         StructField('featuresList', ArrayType(StringType(), True), True),
                         StructField('latitude', DoubleType(), True),
                         StructField('listing_id', LongType(), True),
                         StructField('longitude', DoubleType(), True),
                         StructField('manager_id', StringType(), True),
                         StructField('photos', ArrayType(StringType(), True), True),
                         StructField('price', DoubleType(), True),
                         StructField('street_address', StringType(), True),
                         StructField('label', IntegerType(), True)])

    input_data_df = sqlCtx.createDataFrame(input_data_pd, schema)
    Logger.logger.info(input_data_df.count())
    input_data_df.printSchema()

    input_data_df.show(10)
    # input_data_LabeledData.show(151,truncate=False)
    (train, test) = input_data_df.randomSplit([0.8, 0.2])




    pipelineConfig = PipConfig(finalClassifier)

    Logger.logger.info("pipeline stages used: " + str(pipelineConfig.getStages()))

    crossval = CrossValidator(estimator=Pipeline(stages=pipelineConfig.getStages()),
                              estimatorParamMaps=pipelineConfig.getParamGrid(),
                              evaluator=pipelineConfig.getModelEvaluator(),
                              numFolds=5)

    crossvalModel = crossval.fit(train)
    predictions = crossvalModel.transform(test)
    finalOutput = predictions.select(["listing_id","probability"])
    # predictions.show(10)
    predictions.select(["listing_id","label","probability","prediction"]).show(10, truncate=False)


    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

    accuracy = evaluator.evaluate(predictions)
    Logger.logger.info("Test Error = %g" % (1.0 - accuracy))


    def log_loss(results_transformed, label="label", probability="probability"):
        labs_and_preds = results_transformed[label, probability]
        return - labs_and_preds \
            .rdd \
            .map(lambda x: np.log(x[1][x[0]])) \
            .reduce(lambda x, y: x + y) / labs_and_preds.count()


    Logger.logger.info("Log Loss = %g" % log_loss(predictions.select(["label", "probability"])))

    test_confusion_matrix_pd = predictions.select("label", "prediction").toPandas()
    cnf_matrix = confusion_matrix(test_confusion_matrix_pd["label"], test_confusion_matrix_pd["prediction"])
    Logger.logger.info("Here is the confusion matrix with both row and column indices as: high, medium, low")
    Logger.logger.info(cnf_matrix)
    varImp = crossvalModel.bestModel.stages[-1].featureImportances
    Logger.logger.info("Printing the feature importances")
    Logger.logger.info(varImp)

    predictionsPandas = predictions.select(["listing_id","label","probability","prediction"]).toPandas()
    predictionsPandas.to_csv("predictionOutput.csv")
else:
    Logger.logger.info("Doing the real prediction")
    input_data_pd_train = pd.read_json("data/train.json")
    input_data_pd_train["bedrooms"]+=0.0
    input_data_pd_train["created"] = input_data_pd_train["created"].apply(lambda s: time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()))
    input_data_pd_train["price"] += 0.0

    input_data_pd_test = pd.read_json("data/test.json")
    input_data_pd_test["bedrooms"]+=0.0
    input_data_pd_test["created"] = input_data_pd_test["created"].apply(lambda s: time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()))
    input_data_pd_test["price"] += 0.0


    multiclassLabelAssigner = MulticlassLabelAssigner(["high", "medium", "low"])
    input_data_pd_train["interest_level"] = input_data_pd_train["interest_level"].apply(lambda s: multiclassLabelAssigner.assign(inputTxt=s))

    # input_data_pd[["bedrooms"]] = input_data_pd[["bedrooms"]].apply(pd.to_numeric(downcast='float'))
    schemaTrain = StructType([StructField('bathrooms', DoubleType(), True),
                         StructField('bedrooms', DoubleType(), True),
                         StructField('building_id', StringType(), True),
                         StructField('created', DoubleType(), True),
                         StructField('description', StringType(), True),
                         StructField('display_address', StringType(), True),
                         StructField('featuresList', ArrayType(StringType(), True), True),
                         StructField('latitude', DoubleType(), True),
                         StructField('listing_id', LongType(), True),
                         StructField('longitude', DoubleType(), True),
                         StructField('manager_id', StringType(), True),
                         StructField('photos', ArrayType(StringType(), True), True),
                         StructField('price', DoubleType(), True),
                         StructField('street_address', StringType(), True),
                         StructField('label', IntegerType(), True)])

    schemaTest = StructType([StructField('bathrooms', DoubleType(), True),
                         StructField('bedrooms', DoubleType(), True),
                         StructField('building_id', StringType(), True),
                         StructField('created', DoubleType(), True),
                         StructField('description', StringType(), True),
                         StructField('display_address', StringType(), True),
                         StructField('featuresList', ArrayType(StringType(), True), True),
                         StructField('latitude', DoubleType(), True),
                         StructField('listing_id', LongType(), True),
                         StructField('longitude', DoubleType(), True),
                         StructField('manager_id', StringType(), True),
                         StructField('photos', ArrayType(StringType(), True), True),
                         StructField('price', DoubleType(), True),
                         StructField('street_address', StringType(), True)])

    input_data_df_train = sqlCtx.createDataFrame(input_data_pd_train, schemaTrain)
    input_data_df_test = sqlCtx.createDataFrame(input_data_pd_test, schemaTest)

    input_data_df_train.printSchema()
    input_data_df_test.printSchema()
    input_data_df_train.show(10)

    train = input_data_df_train
    test = input_data_df_test

    logisticR = LogisticRegression(maxIter=20, family="multinomial")#, regParam=0.3, elasticNetParam=0.8)

    pipelineConfig = PipConfig(finalClassifier)

    Logger.logger.info("pipeline stages used: " + str(pipelineConfig.getStages()))

    crossval = CrossValidator(estimator=Pipeline(stages=pipelineConfig.getStages()),
                              estimatorParamMaps=pipelineConfig.getParamGrid(),
                              evaluator=pipelineConfig.getModelEvaluator(),
                              numFolds=5)

    crossvalModel = crossval.fit(train)
    predictions = crossvalModel.transform(test)
    finalOutput = predictions.select(["listing_id", "probability"])
    finalOutput.show(10, truncate=False)
    # finalOutput.printSchema()

    # evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    #
    # accuracy = evaluator.evaluate(predictions)
    # Logger.logger.info("Test Error = %g" % (1.0 - accuracy))



    # trainingSummary = cvModel.bestModel.summary
    #
    # Logger.logger.info(trainingSummary.totalIterations)
    #
    # Logger.logger.info(trainingSummary.objectiveHistory) # one value for each iteration
    #
