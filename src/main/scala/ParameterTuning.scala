import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType

object ParameterTuning {
  def main(args: Array[String]){
    if (args.length != 1){
      println("Usage: ParameterTuning <model_name>\n" +
        "Available models:\n" +
        "\tlogistic - Logistic Regression model\n" +
        "\trandom_forest - Random Forest model")
      System.exit(0)
    }
    // Get spark session
    val spark = SparkSession.builder
      .appName("Spark CSV Reader")
      .getOrCreate

    // Load train data
    var train_df = spark.read
      .format("csv")
      .option("header", "true")
      .load("dataset/train.csv")

    // Load pipeline model
    val pipeline = PipelineModel.load("./pipeline")
    // Get features by transforming the train model
    val features = pipeline.transform(train_df)
    // Get test validation split
    val Array(trainingData, testData) = features.withColumn("label", features("Sentiment").cast(DoubleType)).randomSplit(Array(0.8, 0.2), seed = 1234L)
    // Init the model we will tune (logistic regression, random forest)
    // And their respective parameter grid
    val logistic =  new LogisticRegression()
      .setFeaturesCol("normedW2V")
      .setLabelCol("label")
    val model_name = args(0)
    val paramGridLogistic = new ParamGridBuilder()
      .addGrid(logistic.regParam, Array(0.1, 0.01, 0.3, 1))
      .addGrid(logistic.maxIter, Array(10, 20, 50, 100))
      .addGrid(logistic.elasticNetParam, Array(0, 0.1, 0.5, 0.8))
      .build()

    val rf = new RandomForestClassifier()
      .setFeaturesCol("normedW2V")
      .setLabelCol("label")

    val paramGridRandomForest = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(25, 50, 100, 200))
      .addGrid(rf.maxDepth, Array(5, 20, 50, 75))
      .addGrid(rf.minInstancesPerNode, Array(2,5,8,10))
      .build()

    // Create Cross Validator
    val cv = new CrossValidator()

    if (model_name=="logistic") {
      cv
        .setEstimator(logistic)
        .setEstimatorParamMaps(paramGridLogistic)
      } else if (model_name=="random_forest") {
      cv
        .setEstimator(rf)
        .setEstimatorParamMaps(paramGridRandomForest)
      }

    val cvModel = cv
      .setNumFolds(5)
      .fit(trainingData)

    // Evaluate model and print best one
    val predictions = cvModel.transform(testData).select("label", "SentimentText","prediction")
    val evaluator = new BinaryClassificationEvaluator()
    val accuracy = evaluator.evaluate(predictions.withColumnRenamed("prediction", "rawPrediction"))
    val param = cvModel.bestModel.extractParamMap().toString()
    println(s"$param")
    println(s"Accuracy $accuracy")
  }

}
