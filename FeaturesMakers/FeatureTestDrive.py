from FeaturesMakers.AbstractFeatureProtocol import AbstractFeatureProtocal
from pyspark.sql.dataframe import DataFrame

class FeatureTestDrive(AbstractFeatureProtocal):
    def __init__(self, trainingDataFrame:DataFrame):
        self.trainingDataFrame = trainingDataFrame
        print(trainingDataFrame)
    def create(self):
        return self.trainingDataFrame


# test = FeatureTestDrive(1)
