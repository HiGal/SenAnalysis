import java.util.Calendar

import model.word2vec
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Normalizer, RegexTokenizer, StopWordsRemover, VectorAssembler}
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
    val dT = Calendar.getInstance()
    val currentMinute = dT.get(Calendar.MINUTE)
    val currentHour = dT.get(Calendar.HOUR_OF_DAY)
    val currentDate = dT.get(Calendar.DATE)
    println(s"$currentDate $currentHour:$currentMinute")
    val tokenizer = new RegexTokenizer()
      .setInputCol("SentimentText")
      .setOutputCol("Tokens")
      .setPattern("\\W+")
      .setGaps(true)
    val remover = new StopWordsRemover()
      .setInputCol("Tokens")
      .setOutputCol("filteredPhrases")
    val cleaner = new Cleaner()
      .setInputCol("SentimentText")
      .setOutputCol("SentimentText")
      .setRegularExpressions(Array("@[a-zA-Z0-9_]+", "&(lt)?(gt)?(amp)?(quot)?;"))
    val model = new word2vec().load_model("./word2vec.model").setInputCol("filteredPhrases").setOutputCol("word2vec")
    val normalizer = new Normalizer()
      .setInputCol("word2vec")
      .setOutputCol("normedW2V")
    val streamPipeline = new Pipeline().setStages(Array( cleaner, tokenizer, remover, model, normalizer))
    test_df.show(50)
    val sentenceDataFrame = spark.createDataFrame(Seq(("Hi I heard about Spark", 1))).toDF("SentimentText", "id")
    val a = PipelineModel.load("./pipeline")
    test_df = a.transform(test_df)
    test_df.show(50)
    //    val tmp = tokenizer.transform(train_df).select("Tokens").union(tokenizer.transform(test_df).select("Tokens"))
    //    val model = new word2vec().train(tmp, 50)
    //    model.save("./word2vec.model")
    val tmp = tokenizer.transform(train_df)
    val tmp1 = remover.transform(tmp)
    val w2v = new word2vec().load_model("./w2v_big")
    var result = w2v.transform(tmp1)


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
