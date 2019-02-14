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

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import RegressionEvaluator
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

doTest = str(argsDict.get("doTest")).lower()
if doTest != "true" or doTest != "false":
    doTest = True

spark = SparkSession \
    .builder \
    .master("local") \
    .appName("wisc_breast_cancer_analysis") \
    .config("spark.executor.memory", '8g') \
    .config('spark.executor.cores', '4') \
    .config('spark.cores.max', '4') \
    .config("spark.driver.memory",'8g') \
    .getOrCreate()

sc = SparkContext.getOrCreate()
sqlCtx = SQLContext(sc)

from Miscellaneous.ModelPipConfig import PipConfig


if doTest:
    input_data_pd = pd.read_json("data/train.json")
    input_data_pd["bedrooms"]+=0.0
    input_data_pd["created"] = input_data_pd["created"].apply(lambda s: time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()))
    input_data_pd["price"] += 0.0
    multiclassLabelAssigner = MulticlassLabelAssigner()
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

    logisticR = LogisticRegression(maxIter=20, family="multinomial")#, regParam=0.3, elasticNetParam=0.8)

    modelEvaluator = RegressionEvaluator()
    pipelineConfig = PipConfig()

    pipeline = Pipeline(stages=pipelineConfig.getStages())



    Logger.logger.info("pipeline stages used: " + str(pipelineConfig.getStages()))

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=pipelineConfig.getParamGrid(),
                              evaluator=modelEvaluator,
                              numFolds=3)

    # model = pipeline.fit(train)


    cvModel = crossval.fit(train)
    predictions = cvModel.transform(test)
    predictions.orderBy('bathrooms', ascending=False).show(10, truncate=False)
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

    accuracy = evaluator.evaluate(predictions)
    Logger.logger.info("Test Error = %g" % (1.0 - accuracy))


    # trainingSummary = cvModel.bestModel.summary
    #
    # Logger.logger.info(trainingSummary.totalIterations)
    #
    # Logger.logger.info(trainingSummary.objectiveHistory) # one value for each iteration
    #
