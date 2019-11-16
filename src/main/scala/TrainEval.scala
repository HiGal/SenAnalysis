import model.word2vec
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Normalizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.Column


object TrainEval {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("Spark CSV Reader")
      .getOrCreate
    var train_df = spark.read
      .format("csv")
      .option("header", "true")
      .load("dataset/train.csv")
    var test_df = spark.read
      .format("csv")
      .option("header", "true")
      .load("dataset/test.csv")
    train_df = train_df.withColumn("SentimentText", regexp_replace(train_df("SentimentText"), "@[a-zA-Z0-9_]+", ""))
    train_df = train_df.withColumn("SentimentText", regexp_replace(train_df("SentimentText"), "&([lt][gt][amp][quot]);", " "))
    train_df = train_df.withColumn("Sentiment", train_df("Sentiment").cast(IntegerType))
    test_df = test_df.withColumn("SentimentText", regexp_replace(test_df("SentimentText"), "@[a-zA-Z0-9_]+", ""))
    test_df = test_df.withColumn("SentimentText", regexp_replace(test_df("SentimentText"), "&([lt][gt][amp][quot]);", " "))

    val tokenizer = new RegexTokenizer()
      .setInputCol("SentimentText")
      .setOutputCol("token")
      .setPattern("\\W+")
      .setGaps(true)

    val remover = new StopWordsRemover()
      .setInputCol("token")
      .setOutputCol("filteredPhrases")


    //    val tmp = tokenizer.transform(train_df).select("Tokens").union(tokenizer.transform(test_df).select("Tokens"))
    //    val model = new word2vec().train(tmp, 50)
    //    model.save("./word2vec.model")
    val tmp = tokenizer.transform(train_df)
    val tmp1 = remover.transform(tmp)
    val w2v = new word2vec().load_model("./w2v_big")
    var result = w2v.transform(tmp1)

    val normalizer = new Normalizer()
      .setInputCol("word2vec")
      .setOutputCol("normedW2V")

    result = normalizer.transform(result)
    result.write.format("csv").save("normed.csv")

    val Array(trainingData, testData) = result.randomSplit(Array(0.8, 0.2), seed = 1234L)

    val logistic = new LogisticRegression()
      .setMaxIter(100)
      .setFeaturesCol("normedW2V")
      .setLabelCol("Sentiment")
      .fit(trainingData)


    val test_features = normalizer.transform(w2v.transform(remover.transform(tokenizer.transform(test_df))))
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
    //    println(s"Count 1 =$ \nCount 2 =$")
  }

}
