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
    (1.0, float("nan")),
    (2.0, float("nan")),
    (4.0, 3.0),
    (4.0, 4.0),
    (5.0, 5.0)
], ["a", "b"])

imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])

pip = Pipeline(stages=[imputer])

model = pip.fit(df)

model.transform(df).show()