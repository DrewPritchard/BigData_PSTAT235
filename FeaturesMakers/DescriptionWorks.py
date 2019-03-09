from pyspark.sql.functions import isnan
from pyspark.sql.functions import when, lit, col
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline, Transformer


class DescriptionWorks(Transformer):
    def __init__(self, k: int, inputCol: str = None, outputCol: str = None):
        super(DescriptionWorks, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.k = int(k)

    # Make the replace function
    @staticmethod
    def replace(column, value):
        return when(column != value, column).otherwise(lit("none"))

    # Make the full function
    def _transform(self, df):
        # select only the description
        #train_data_df2 = df.select("interest_level","description")
        # clean blanks
        train4 = df.withColumn(self.inputCol, DescriptionWorks.replace(col(self.inputCol), '        '))
        train4 = train4.withColumn(self.inputCol, DescriptionWorks.replace(col(self.inputCol), ""))
        train4 = train4.withColumn(self.inputCol, DescriptionWorks.replace(col(self.inputCol), " "))
        train4 = train4.withColumn(self.inputCol, DescriptionWorks.replace(col(self.inputCol), "           "))
        # regular expression tokenizer
        regexTokenizer = RegexTokenizer(inputCol="description", outputCol="words", pattern="\\W") # I don't know what W is...

        # stop words
        add_stopwords = ["a","the","it","of","the","is","and", # standard stop words
         "A","this","in","for"]
        stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)

        # count vectorizer
        countVectors = CountVectorizer(inputCol="filtered", outputCol=self.outputCol, vocabSize=self.k, minDF=5)

        pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors])

        # Fit the pipeline to training documents.
        pipelineFit = pipeline.fit(train4)
        dataset = pipelineFit.transform(train4)
        # dataset = dataset.withColumn("label", dataset["interest_level"].cast(IntegerType()))
        dataset = dataset.drop("words")
        dataset = dataset.drop("filtered")

        #return dataset["word_features"]
        return dataset
