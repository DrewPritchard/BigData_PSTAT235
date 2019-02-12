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

input_data_pd = pd.read_json("toyData/iris.json")
input_data_df = sqlCtx.createDataFrame(input_data_pd)

# print(input_data_df)



multiclassLabelAssigner = MulticlassLabelAssigner()
f_to_float64 = lambda x: np.array(x).astype(np.float64)
input_data_PipelinedRDD = input_data_df.rdd.map(lambda x: (multiclassLabelAssigner.assign(x[-1]),
                                                           DenseVector(f_to_float64(x[:-1]))))
input_data_LabeledData = sqlCtx.createDataFrame(input_data_PipelinedRDD, ["label", "features"])
input_data_LabeledData.printSchema()
# input_data_LabeledData.show(151,truncate=False)
(train, test) = input_data_LabeledData.randomSplit([0.8, 0.2])
logisticR = LogisticRegression(maxIter=20, family="multinomial")#, regParam=0.3, elasticNetParam=0.8)


modelEvaluator = RegressionEvaluator()
pipeline = Pipeline(stages=[logisticR])


paramGrid = ParamGridBuilder().addGrid(logisticR.regParam,
                                       [0.1, 0.01]).addGrid(logisticR.elasticNetParam,
                                                            [0, 1]).build()

# ovr = OneVsRest(classifier=logisticR)


crossval = CrossValidator(estimator=logisticR,
                          estimatorParamMaps=paramGrid,
                          evaluator=modelEvaluator,
                          numFolds=3)

print(train.take(1)[0]["features"])

cvModel = crossval.fit(train)
predictions = cvModel.transform(test)
predictions.show(10, truncate=False)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
Logger.logger.info("Test Error = %g" % (1.0 - accuracy))


# trainingSummary = cvModel.bestModel.summary
#
# Logger.logger.info(trainingSummary.totalIterations)
#
# Logger.logger.info(trainingSummary.objectiveHistory) # one value for each iteration
#
