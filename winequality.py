
#Loading the libraries
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import numpy as np
import sys


#Starting the spark session
conf = pyspark.SparkConf().setAppName('winequality')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

#Loading the dataset
path = sys.argv[1]    
df = spark.read.format("csv").load(path , inferSchema='true',header = True ,sep =";")
df.printSchema()
df.show()
#changing the 'quality' column name to 'label'
for col_name in df.columns[1:-1]+['""""quality"""""']:
    df = df.withColumn(col_name, col(col_name).cast('float'))
df = df.withColumnRenamed('""""quality"""""', "label")

# Convert to float format
def string_to_float(x):
    return float(x)

#catelogy the data
def catelogy(r):
    if (0<= r <= 6.5):
        label = "bad"
    elif(6.5 < r <= 10):
        label = "good"
    else:
        label = "n/a"
    return label

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, DoubleType

string_to_float_udf = udf(string_to_float, DoubleType())
quality_udf = udf(lambda x: catelogy(x), StringType())

df = df.withColumn("label", quality_udf("label"))
df.show(5)
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString,StringIndexer, VectorIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

transformed = transData(df)

labelIndexer = StringIndexer(inputCol='label',
                             outputCol='indexedLabel').fit(transformed)
labelIndexer.transform(transformed)

featureIndexer =VectorIndexer(inputCol="features", \
                                  outputCol="indexedFeatures", \
                                  maxCategories=4).fit(transformed)
featureIndexer.transform(transformed)

#TODO
from pyspark.ml.feature import PCA
data = transformed
pca = PCA(k=6, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(data)

result = model.transform(data).select("pcaFeatures")

(trainingData, testData) = transformed.randomSplit([0.8, 0.2])

from pyspark.ml.classification import RandomForestClassifier

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=8,maxDepth=20,seed=42)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", 
                               outputCol="predictedLabel",
                               labels=labelIndexer.labels)
# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf,labelConverter])
# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)
# Select example rows to display.



from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", 
                                              predictionCol="prediction", 
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("accuracy = %g" % (accuracy))

rfModel = model.stages[-2]
model.save(sys.argv[2])
print('Trained model is save to path %s' %(sys.argv[2]))
