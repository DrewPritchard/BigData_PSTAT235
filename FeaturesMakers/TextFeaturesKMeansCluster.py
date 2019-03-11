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
from pyspark.ml.feature import VectorAssembler


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
        feat_df = df.withColumn("features_list", F.split(df[inputCol], ',| \* '))
        #explodes the features into column "ex_features_list"
        feat_df_ex = feat_df.withColumn("ex_features_list", F.explode(feat_df["features_list"]))
        #creates clustering data frame with only column "ex_features_list"
        clustering_df = feat_df_ex[["ex_features_list"]]
        #renames the column
        clustering_df = clustering_df.withColumnRenamed("ex_features_list", "text")

        #creates a tokenizer
        tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
        #removes stop words
        remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
        #hashes the features into sparse vectors
        hashingTF = HashingTF(inputCol="stopWordsRemovedTokens", outputCol="rawFeatures", numFeatures=2000)
        #invers document frequency - importance of the work (kind of)
        idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)

        #creates and fits the pipeline
        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
        pipelined_df = pipeline.fit(clustering_df).transform(clustering_df)

        #Set the number of clusters determined in rentalPrice_jonas.ipynb
        num_k = 18
        #creates the k-means
        km = BisectingKMeans(k = num_k)
        #fits it to the pipelined data frame
        model = km.fit(pipelined_df)
        #transform into the results
        results = model.transform(pipelined_df)
        #changes the name of the column "prediction" to "cluster"
        results = results.withColumnRenamed("prediction", "clusters")

        join_df = results.drop(*["tokens", "stopWordsRemovedTokens", "rawFeatures", "features"])
        #creates a column to add on\n",
        join_df = join_df.withColumn("join_col", F.monotonically_increasing_id())
        feat_df_ex = feat_df_ex.withColumn("join_col", F.monotonically_increasing_id())
        #joins the df_together\n",
        joined_df = feat_df_ex.join(join_df, feat_df_ex["join_col"] == join_df["join_col"], how = "left")
        joined_df = joined_df.drop("join_col")
        #have to ad constatnt column for the pivot function
        joined_df = joined_df.withColumn("constant_val", F.lit(1))
        #pivots the data frame
        df_piv = joined_df\
                       .groupBy("listing_id")\
                       .pivot("clusters")\
                       .agg(F.coalesce(F.first("constant_val")))
        #Joins the data frame to the original\n",
        df = df.join(df_piv, on = "listing_id", how = "left")
        #store the colusters in list, removes "listing_id"
        cluster_col = df_piv.columns
        cluster_col.remove("listing_id")
        #fills missing values",
        df = df.fillna(0, subset = cluster_col)
        #changes the names of the columns to "#_feature_cluster" to the stings",

        va = VectorAssembler(inputCols=[x for x in cluster_col], outputCol=self.outputCol)
        df = va.transform(df)
        for c in cluster_col:
            df = df.drop(c)
        return df
