import model.word2vec
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{LabeledPoint, RegexTokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.types.{DoubleType, IntegerType}

object ParameterTuning {
  def main(args: Array[String]){

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("Spark CSV Reader")
      .getOrCreate
    var train_df = spark.read
      .format("csv")
      .option("header", "true")
      .load("dataset/train.csv")

    val pipeline = PipelineModel.load("./pipeline")
    val features = pipeline.transform(train_df)
    val Array(trainingData, testData) = features.withColumn("label", features("Sentiment").cast(DoubleType)).randomSplit(Array(0.8, 0.2), seed = 1234L)
    val logistic =  new LogisticRegression()
      .setFeaturesCol("result")
      .setLabelCol("label")

    val paramGridLogistic = new ParamGridBuilder()
      .addGrid(logistic.regParam, Array(0.1, 0.01, 0.3, 1))
      .addGrid(logistic.maxIter, Array(10, 20, 50, 100))
      .addGrid(logistic.elasticNetParam, Array(0, 0.1, 0.5, 0.8))
      .build()
    val cv = new CrossValidator()
      .setEstimator(logistic)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGridLogistic)
      .setNumFolds(5)
    val cvModel = cv.fit(trainingData)

    val predictions = cvModel.transform(testData).select("label", "SentimentText","prediction")

    val evaluator = new BinaryClassificationEvaluator()
    val accuracy = evaluator.evaluate(predictions.withColumnRenamed("prediction", "rawPrediction"))
    val param = cvModel.bestModel.extractParamMap().toString()
    println(s"$param")
    println(s"Accuracy $accuracy")
  }

}
