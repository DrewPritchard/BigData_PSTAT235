from abc import ABCMeta, abstractmethod, ABC

from pyspark.sql.dataframe import DataFrame

class AbstractFeatureProtocal(ABC):
    @abstractmethod
    def __init__(self, trainingDataFrame: DataFrame):
        pass

    @abstractmethod
    def create(self):
        pass