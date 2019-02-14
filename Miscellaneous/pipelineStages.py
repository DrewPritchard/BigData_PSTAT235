from pyspark.ml.feature import Imputer
from pyspark.ml.classification import LogisticRegression, OneVsRest
from FeaturesMakers import *
from pyspark.ml.feature import VectorAssembler

# allfea = ["bathrooms",
#           "bedrooms",
#           "building_id",
#           "created",
#           "description",
#           "display_address",
#           "features",
#           "latitude",
#           "listing_id",
#           "longitude",
#           "manager_id",
#           "photos",
#           "price",
#           "interest_level"]


#
# ignore = ["building_id",
#           "created",
#           "description",
#           "display_address",
#           "features",
#           "latitude",
#           "listing_id",
#           "longitude",
#           "manager_id",
#           "photos"]


imputer = Imputer(strategy="mean",
                  inputCols=["bathrooms", "bedrooms"],
                  outputCols=["out_bathrooms", "out_bedrooms"])

assembler = VectorAssembler(inputCols=["out_bathrooms", "out_bedrooms"], outputCol="features")

logisticR = LogisticRegression(maxIter=20, family="multinomial")  # , regParam=0.3, elasticNetParam=0.8)

stages = [imputer, assembler, logisticR]

