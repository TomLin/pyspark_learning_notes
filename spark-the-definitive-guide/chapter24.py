from pyspark.ml.linalg import Vectors
denseVec = Vectors.dense(1.0, 2.0, 3.0)
size = 3
idx = [1, 2] # locations of non-zero elements in vector
values = [2.0, 3.0]
sparseVec = Vectors.sparse(size, idx, values)


# COMMAND ----------

# Start a SparkSession -> ref: https://spark.apache.org/docs/2.3.0/sql-programming-guide.html#starting-point-sparksession
from pyspark.sql import SparkSession

spark = SparkSession\
  .builder\
  .appName('MLlib Examples')\
  .getOrCreate()


df = spark.read.json("/databricks-datasets/definitive-guide/data/simple-ml")
df.orderBy("value2").show()

# +-----+----+------+------------------+
# |color| lab|value1|            value2|
# +-----+----+------+------------------+
# |green|good|     1|14.386294994851129|
# |green| bad|    16|14.386294994851129|
# | blue| bad|     8|14.386294994851129|
# | blue| bad|     8|14.386294994851129|
# | blue| bad|    12|14.386294994851129|
# |green| bad|    16|14.386294994851129|
# |green|good|    12|14.386294994851129|
# |  red|good|    35|14.386294994851129|
# |  red|good|    35|14.386294994851129|


# COMMAND ----------

# When we use MLlib, all inputs to machine learning algorithms
# (with several exceptions discussed in later chapters)
# in Spark must consist of type Double (for labels) and Vector[Double] (for features).

from pyspark.ml.feature import RFormula
supervised = RFormula(formula="lab ~ . + color:value1 + color:value2")

# RFormula will automatically handle categorical variables for us.

# COMMAND ----------

fittedRF = supervised.fit(df) # need to fit the data first, and then transform the data
preparedDF = fittedRF.transform(df)
preparedDF.show()

# RFormula inspects our data during the fit call and outputs an object that will transform our data
# according to the specified formula, which is called an RFormulaModel .
# When we use this transformer, Spark automatically converts our categorical variable to Doubles
# so that we can input it into a (yet to be specified) machine learning model.
# In particular, it assigns a numerical value to each possible color category,
# creates additional features for the interaction variables between colors and value1/value2,
# and puts them all into a single vector.


# +-----+----+------+------------------+--------------------+-----+
# |color| lab|value1|            value2|            features|label|
# +-----+----+------+------------------+--------------------+-----+
# |green|good|     1|14.386294994851129|(10,[1,2,3,5,8],[...|  1.0|
# | blue| bad|     8|14.386294994851129|(10,[2,3,6,9],[8....|  0.0|
# | blue| bad|    12|14.386294994851129|(10,[2,3,6,9],[12...|  0.0|
# |green|good|    15| 38.97187133755819|(10,[1,2,3,5,8],[...|  1.0|
# |green|good|    12|14.386294994851129|(10,[1,2,3,5,8],[...|  1.0|
# |green| bad|    16|14.386294994851129|(10,[1,2,3,5,8],[...|  0.0|
# |  red|good|    35|14.386294994851129|(10,[0,2,3,4,7],[...|  1.0|



# COMMAND ----------

train, test = preparedDF.randomSplit([0.7, 0.3])


# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="label",featuresCol="features")


# COMMAND ----------

print(lr.explainParams()) # explain hyperparameters and their default values

# elasticNetParam: the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty. (default: 0.0)
# family: The name of family which is a description of the label distribution to be used in the model. Supported options: auto, binomial, multinomial (default: auto)
# featuresCol: features column name. (default: features, current: features)
# fitIntercept: whether to fit an intercept term. (default: True)
# labelCol: label column name. (default: label, current: label)
# lowerBoundsOnCoefficients: The lower bounds on coefficients if fitting under bound constrained optimization. The bound matrix must be compatible with the shape (1, number of features) for binomial regression, or (number of classes, number of features) for multinomial regression. (undefined)
# lowerBoundsOnIntercepts: The lower bounds on intercepts if fitting under bound constrained optimization. The bounds vector size must beequal with 1 for binomial regression, or the number oflasses for multinomial regression. (undefined)
# maxIter: max number of iterations (>= 0). (default: 100)
# predictionCol: prediction column name. (default: prediction)
# regParam: regularization parameter (>= 0). (default: 0.0)


# COMMAND ----------

fittedLR = lr.fit(train)

# We make predictions with the transform method.
# For example, we can transform our training dataset to see
# what labels our model assigned to the training data and
# how those compare to the true outputs.
# This, again, is just another DataFrame we can manipulate.

fittedLR.transform(train).select("label", "prediction").show() # transform() will append prediction col in the original dataframe

# +-----+----------+
# |label|prediction|
# +-----+----------+
# |  0.0|       0.0|
# |  0.0|       0.0|
# |  0.0|       0.0|
# |  0.0|       0.0|
# |  0.0|       0.0|
# |  0.0|       0.0|
# |  0.0|       0.0|





#---------- PIPELINE OUR WORKFLOW ----------#


# COMMAND ----------

train, test = df.randomSplit([0.7, 0.3])


# COMMAND ----------

rForm = RFormula()
lr = LogisticRegression().setLabelCol("label").setFeaturesCol("features") # in model, always use 'set' to specify columns


# COMMAND ----------

from pyspark.ml import Pipeline
stages = [rForm, lr]
pipeline = Pipeline().setStages(stages)


# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder
params = ParamGridBuilder()\
  .addGrid(rForm.formula, [
    "lab ~ . + color:value1",
    "lab ~ . + color:value1 + color:value2"])\
  .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
  .addGrid(lr.regParam, [0.1, 2.0])\
  .build()


# COMMAND ----------

#---------- EVALUATOR ----------#

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()\
  .setMetricName("areaUnderROC")\
  .setRawPredictionCol("prediction")\
  .setLabelCol("label")


# COMMAND ----------


#---------- Validation Dataset ----------#
# Either 'TrainValidationSplit' or 'CrossValidator'
from pyspark.ml.tuning import TrainValidationSplit
tvs = TrainValidationSplit()\
  .setTrainRatio(0.75)\
  .setEstimatorParamMaps(params)\
  .setEstimator(pipeline)\
  .setEvaluator(evaluator)


# Let’s run the entire pipeline we constructed.
# To review, running this pipeline will test out every version of the model against the validation set.
# Note the type of tvsFitted is TrainValidationSplitModel.

# COMMAND ----------

tvsFitted = tvs.fit(train)

# COMMAND ----------

# And of course evaluate how it performs on the test set!
evaluator.evaluate(tvsFitted.transform(test))

# Get the best model params: https://stackoverflow.com/questions/41781529/how-to-print-best-model-params-in-pyspark-pipeline


#---------- Persist and Apply Model ----------#
tvsFitted.write.overwrite().save('./Folder')

# After writing out the model, we can load it into another Spark program to make predictions.
# To do this, we need to use a “model” version of our particular algorithm to
# load our persisted model from disk. If we were to use CrossValidator,
# we’d have to read in the persisted version as the CrossValidatorModel,
# and if we were to use LogisticRegression manually we would have to use LogisticRegressionModel.
# In this case, we use TrainValidationSplit, which outputs TrainValidationSplitModel.
