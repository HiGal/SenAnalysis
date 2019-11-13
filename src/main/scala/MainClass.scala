import model.word2vec
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.log4j.Level
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.ml.feature.{RegexTokenizer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.types.IntegerType

object MainClass{
  def main(args: Array[String]){

    val spark = SparkSession.builder
      .master("local")
      .appName("Spark CSV Reader")
      .getOrCreate

    var df = spark.read
      .format("csv")
      .option("header", "true")
      .load("dataset/train.csv")
    df = df.withColumn("SentimentText", regexp_replace(df("SentimentText"), "@[a-zA-Z0-9_]+", "") )
    df = df.withColumn("Sentiment", df("Sentiment").cast(IntegerType))
    val tokenizer = new RegexTokenizer()
      .setInputCol("SentimentText")
      .setOutputCol("Tokens")
      .setPattern("\\W+")
      .setGaps(true)

    val tmp = tokenizer.transform(df)

    val model = new word2vec().train(tmp, 1)
    val result = model.transform(tmp)
    val Array(trainingData, testData) = result.randomSplit(Array(0.8, 0.2), seed = 1234L)
    val logistic =  new LogisticRegression()
      .setMaxIter(1)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFeaturesCol("result")
      .setLabelCol("Sentiment")
      .fit(trainingData)

    val test_df = spark.read
      .format("csv")
      .option("header", "true")
      .load("dataset/test.csv")
    val test_features = model.transform(tokenizer.transform(test_df))
    var predictions = logistic.transform(test_features)
    println("Predictions on test data")
    predictions.show()

    println("Predictions on validation set")
    predictions = logistic.transform(testData)
    predictions.show()

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Sentiment")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Validation set accuracy = $accuracy")
  }
}