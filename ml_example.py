"""
Below is the code to train a binary classification machine learning model using Spark ML for detecting heart disease of an individual based on the given features.
The label of the dataset to predict is the 'target' column

Two Decision Tree models will be created. The first one is a simple DT model and the second model is a result of tuning using the CrossValiation (5-fold in this example).
Models are evaluated based on the AUC score.
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


def build_pipeline_stages(df, label_col):
  feature_cols = [c for c in df.columns if c != label_col]  # Here all columns are numerical so
  
  stages_for_pipeline = []
  
  #  Convert label into label indices using the StringIndexer
  label_string_index = StringIndexer(inputCol=label_col, outputCol='label')
  stages_for_pipeline += [label_string_index]

  #  Transform all features into a vector using VectorAssembler
  assembler_inputs = feature_cols
  assembler = VectorAssembler(inputCols=assembler_inputs, outputCol='features')
  stages_for_pipeline += [assembler]
  
  return stages_for_pipeline
  

def prepare_training_data(dataset, stages):
  pipeline = Pipeline().setStages(stages)
  pipeline_model = pipeline.fit(dataset)
  return pipeline_model.transform(dataset)


df = spark.table('heart_csv')

# Prepare the dataset for using the pipeline model and keep relevant columns
pipeline_stages = build_pipeline_stages(df, 'target')
dataset = prepare_training_data(df, pipeline_stages)
selected_columns = ['label', 'features'] + df.columns
dataset = dataset.select(selected_columns)


# Randomly split dataframe into training and test sets. Using 80-20 split for train-test data. Setting seed to retain split results
train, test = dataset.randomSplit([0.8, 0.2], seed=100)

# Create initial Decision Tree Model. Setting max depth to 3
decision_tree = DecisionTreeClassifier(labelCol='label', featuresCol='features', maxDepth=3)

# Train first model with train set
decision_tree_model = decision_tree.fit(train)

# Make predictions on test data using the Transformer.transform() method.
predictions_1 = decision_tree_model.transform(test)

# View model's predictions and probabilities of each prediction class
# predictions_1.select('label', 'prediction', 'probability', 'age', 'sex').show()

# Evaluate model using Binary classification evaluator
evaluator = BinaryClassificationEvaluator()

auc_1 = evaluator.evaluate(predictions_1, {evaluator.metricName: 'areaUnderROC'})
print('Area under the curve before tuning: {:.3f}'.format(auc_1))


# Use cross validation to select the best model.
param_grid = (ParamGridBuilder()
         .addGrid(decision_tree.maxDepth, [1, 3, 6, 10])
         .addGrid(decision_tree.maxBins, [20, 40, 80])
         .build())

# Create k-fold CrossValidator. Change this variable to adjust the number of folds, I am using k=5 to for faster grid searching
n_folds = 5
cv = CrossValidator(estimator=decision_tree, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=n_folds)

# Build cross validations
cv_model = cv.fit(train)

# cvModel uses the best model found from the Cross Validation
# Use test set to measure the accuracy of our model on new data
predictions_2 = cv_model.transform(test)

# Evaluate best model
evaluator.evaluate(predictions_2)

# View Best model's predictions and probabilities of each prediction class
# predictions_2.select("label", "prediction", "probability", "age", "sex").show()

auc_2 = evaluator.evaluate(predictions_2, {evaluator.metricName: 'areaUnderROC'})
print('Area under the curve after tuning: {:.3f}'.format(auc_2))
