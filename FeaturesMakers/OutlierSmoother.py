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

class OutlierSmoother(Transformer):
    def __init__(self, thresh: float, inputCols: List[str]=None, outputCols: List[str]=None):
        if len(inputCols) != len(outputCols):
            raise ValueError("the length of input cols must be equal to the length of output cols")
        super(OutlierSmoother, self).__init__()
        self.inputCols = inputCols
        self.outputCols = outputCols
        self.thresh = thresh

    @staticmethod
    def is_outlier(points, thresh=3.5):
        """
        Returns a boolean array with True if points are outliers and False
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        """
        if len(points.shape) == 1:
            points = points[:, None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    @staticmethod
    def replaceToNull(column, value):
        return when(column != value, column).otherwise(lit(None))

    def _transform(self, df: DataFrame) -> DataFrame:
        tmpColList = None
        tmpColListNpArrFinal = None
        sc = SparkContext.getOrCreate()
        new_df = None
        outlierSmotherRowIdColName = "outlierSmotherRowId"
        if outlierSmotherRowIdColName not in df.schema.names:
            df = df.withColumn(outlierSmotherRowIdColName, monotonically_increasing_id())
        else:
            outlierSmotherRowIdColName = "outlierSmotherRowId4" + "_".join(df.schema.names)

        for j in range(len(self.inputCols)):
            tmpColList = df.select([self.inputCols[j]]).rdd.map(lambda r: r[0]).collect()
            tmpColListNpArr = np.array(tmpColList)
            outliersQList = self.is_outlier(tmpColListNpArr, self.thresh)

            Logger.logger.info(str(sum(outliersQList)) + " outliers in column \'" + self.inputCols[j] + "\' has been smoothened")

            outlierIndices = np.where(outliersQList)[0]
            notOutliersQList = np.array([not j for j in outliersQList])


            tmpColListAvg = sum(tmpColListNpArr*notOutliersQList)/sum(notOutliersQList)

            for outlierIndex in outlierIndices:
                tmpColListNpArr[outlierIndex] = tmpColListAvg

            tmpColListNpArrFinal = tmpColListNpArr.tolist()
            tmpColListNpArrDict = dict()

            for arrItem in range(len(tmpColListNpArrFinal)):
                tmpColListNpArrDict[arrItem] = tmpColListNpArrFinal[arrItem]

            def valueToCategory(value):
                return tmpColListNpArrDict.get(value)

            udfValueToCategory = udf(valueToCategory, FloatType())
            df = df.withColumn(self.outputCols[j], udfValueToCategory(outlierSmotherRowIdColName))

        df = df.drop(outlierSmotherRowIdColName)
        return df

