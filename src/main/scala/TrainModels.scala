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


object TrainModels {
  def main(args: Array[String]) {

    // Create spark Session
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("Spark CSV Reader")
      .getOrCreate

    // Load training dataset
    var train_df = spark.read
      .format("csv")
      .option("header", "true")
      .load("dataset/train.csv")

    // Load test dataset
    var test_df = spark.read
      .format("csv")
      .option("header", "true")
      .load("dataset/test.csv")

    // Init tokenizer transformer
    val tokenizer = new RegexTokenizer()
      .setInputCol("SentimentText")
      .setOutputCol("Tokens")
      .setPattern("\\W+")
      .setGaps(true)

    // Init stop words remover transformer (english by default)
    val remover = new StopWordsRemover()
      .setInputCol("Tokens")
      .setOutputCol("filteredPhrases")

    // Ini cleaner transformer that will remove nickname tags and html tags
    val cleaner = new Cleaner()
      .setInputCol("SentimentText")
      .setOutputCol("SentimentText")
      .setRegularExpressions(Array("@[a-zA-Z0-9_]+", "&(lt)?(gt)?(amp)?(quot)?;"))

    // Normalizer model
    val normalizer = new Normalizer()
      .setInputCol("word2vec")
      .setOutputCol("normedW2V")
    // Load word2vec model
    val model = new word2vec().load_model("./word2vec.model").setInputCol("filteredPhrases").setOutputCol("word2vec")

    // Create pipeline - clean the message first, tokenize the lines, remove stop words, run through word2vec and normalize result
    val streamPipeline = new Pipeline().setStages(Array(cleaner, tokenizer, remover, model, normalizer))
    val sentenceDataFrame = spark.createDataFrame(Seq(("Hi I heard about Spark", 1))).toDF("SentimentText", "id")
    val pipeline = PipelineModel.load("./pipeline")
    var result = pipeline.transform(train_df)
    result = result.withColumn("Sentiment", result("Sentiment").cast(IntegerType))

    // Split training set to test and train
    val Array(trainingData, testData) = result.randomSplit(Array(0.8, 0.2), seed = 1234L)

    // Train logistic model
    val logistic = new LogisticRegression()
      .setMaxIter(50)
      .setRegParam(0)
      .setElasticNetParam(0)
      .setFeaturesCol("normedW2V")
      .setLabelCol("Sentiment")
      .fit(trainingData)
    logistic.save("logistic.model")

    println("Predictions on validation set")
    val predictions = logistic.transform(testData)
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
