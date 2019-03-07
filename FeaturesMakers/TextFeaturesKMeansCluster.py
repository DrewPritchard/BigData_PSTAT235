from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaParams, JavaTransformer, _jvm
from pyspark.ml.common import inherit_doc
from pyspark import SparkContext
from pyspark.sql import DataFrame

from pyspark.ml.param.shared import *
from pyspark.ml import Pipeline, Transformer
from typing import Iterable,List
import numpy as np
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.functions import when, lit, col
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import FloatType
from Miscellaneous.Logger import Logger


from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer, StopWordsRemover
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.clustering import BisectingKMeans
import pyspark.sql.types as typ
import pyspark.sql.functions as F


class TextFeaturesKMeansCluster(Transformer):
    def __init__(self, k: int, inputCol: str = None, outputCol: str = None):
        super(TextFeaturesKMeansCluster, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.k = int(k)

    def _transform(self, df):
        inputCol = self.inputCol
        #creates 1 sting of the features
        string_assembler = F.UserDefinedFunction(lambda x: ','.join(x), typ.StringType())
        df = df.withColumn(inputCol, string_assembler(df[inputCol]))
        #lower case everything
        df = df.withColumn(inputCol, F.lower(df[inputCol]))
        #adds feature "missing features" to NaN
        df = df.withColumn(inputCol,
                                 F.when(df[inputCol] == '', 'missing features')
                                 .otherwise(df[inputCol]))
        #split df on "," and "*" stores as new data frame
        df = df.withColumn("features_list", F.split(df[inputCol], ',| \* '))
        #explodes the features into column "ex_features_list"
        df = df.withColumn("ex_features_list", F.explode(df["features_list"]))
        #creates clustering data frame with only column "ex_features_list"
        #renames the column
        df = df.withColumnRenamed("ex_features_list", "text")

        #creates a tokenizer
        tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
        #removes stop words
        remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
        #hashes the features into sparse vectors
        hashingTF = HashingTF(inputCol="stopWordsRemovedTokens", outputCol="rawFeatures", numFeatures=2000)
        #invers document frequency - importance of the work (kind of)
        idf = IDF(inputCol="rawFeatures", outputCol="out_features", minDocFreq=5)

        #creates and fits the pipeline
        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
        df = pipeline.fit(df).transform(df)

        #Set the number of clusters determined in rentalPrice_jonas.ipynb
        num_k = self.k
        #creates the k-means
        km = BisectingKMeans(k = num_k,featuresCol = "out_features")
        #fits it to the pipelined data frame
        model = km.fit(df)
        #transform into the results
        results = model.transform(df)
        #changes the name of the column "prediction" to "cluster"
        results = results.withColumnRenamed("prediction", self.outputCol)

        return results
