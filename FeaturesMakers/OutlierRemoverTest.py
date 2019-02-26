from FeaturesMakers.OutlierSmoother import OutlierSmoother

from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer
from pyspark.ml import Pipeline

spark = SparkSession \
    .builder \
    .master("local") \
    .appName("wisc_breast_cancer_analysis") \
    .config("spark.executor.memory", '8g') \
    .config('spark.executor.cores', '4') \
    .config('spark.cores.max', '4') \
    .config("spark.driver.memory",'8g') \
    .getOrCreate()


df = spark.createDataFrame([
    (-100.0, -1000.0),
    (2.0, 0.0),
    (4.0, 3.0),
    (4.0, 3.0),
    (4.0, 3.7),
    (4.0, 3.3),
    (4.0, 3.0),
    (4.0, 3.2),
    (4.0, 3.0),
    (4.0, 3.1),
    (4.0, 3.0),
    (5.0, 5.0)], ["a", "b"])

outrem = OutlierSmoother(thresh=3, inputCols=["a", "b"], outputCols=["out_a", "out_b"])

pip = Pipeline(stages=[outrem])

model = pip.fit(df)

model.transform(df).show()

print(outrem)

