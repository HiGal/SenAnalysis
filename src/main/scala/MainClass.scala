import model.word2vec
import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.log4j.Logger
import org.apache.spark.streaming.StreamingContext._
import org.apache.log4j.Level
import org.apache.spark
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{Column, Row, SaveMode, SparkSession}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{RegexTokenizer, VectorAssembler}
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._

object MainClass {
  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .master("local")
      .appName("Spark CSV Reader")
      .getOrCreate
    var df = spark.read
      .format("csv")
      .option("header", "true")
      .load("dataset/train.csv")
    df = df.withColumn("SentimentText", regexp_replace(df("SentimentText"), "@[a-zA-Z0-9_]+", ""))
    df = df.withColumn("Sentiment", df("Sentiment").cast(IntegerType))
    val tokenizer = new RegexTokenizer()
      .setInputCol("SentimentText")
      .setOutputCol("Tokens")
      .setPattern("\\W+")
      .setGaps(true)

    val tmp = tokenizer.transform(df)

    val model = new word2vec().train(tmp, 1)
    model.save("./word2vec.model")

//    val model = new word2vec().load_model("./word2vec.model")
    val result = model.transform(tmp)
    val Array(trainingData, testData) = result.randomSplit(Array(0.8, 0.2), seed = 1234L)
    val logistic = new LogisticRegression()
      .setMaxIter(1)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFeaturesCol("result")
      .setLabelCol("Sentiment")
      .fit(trainingData)


//    val wordSchema = new StructType().add("SentimentText", "string")
//    val socketDF = spark.readStream
//      .format("socket")
//      .option("host", "10.90.138.32")
//      .option("port", 8989)
//      .load()
//      .withColumnRenamed("value", "SentimentText")
//    //      .withColumn("SentimentText", regexp_replace(df("SentimentText"), "@[a-zA-Z0-9_]+", ""))


    val ssc = new StreamingContext(spark.sparkContext, Seconds(1))

    // Words  reading
    val lines = ssc.socketTextStream("10.91.66.168", 8989)
    val schema = new StructType()
      .add(StructField("SentimentText", StringType, true))
    lines.foreachRDD(rdd => {
      val df = spark.createDataFrame(rdd.map(attributes => Row(attributes)), schema)

      val test_features = model.transform(tokenizer.transform(df))
      var predictions = logistic.transform(test_features).select("SentimentText", "prediction")
      predictions.write.mode(SaveMode.Append).csv("./piska_pirata")
    })

//    val query = predictions.writeStream
//      .format("csv")
//      .option("path", "./data/tweets")
//      .option("checkpointLocation", "./checkpoint_path")
//      .start()

//    val allfiles =  spark.read.option("header","false").csv("./data/tweets/part-*.csv")
//    allfiles.coalesce(1).write.format("csv").option("header", "false").save("/data/single_csv.csv/")


    ssc.start()
    ssc.awaitTermination()










    //    // Words  reading
    //    case class Test(id:String, result:String)
    //    val lines = ssc.socketTextStream("10.91.66.168", 8989)
    //    val df1 = spark.createDataFrame(lines).toDF()
    //    val results = model.transform(tmp)


    //    val test_df = spark.read
    //      .format("csv")
    //      .option("header", "true")
    //      .load("dataset/test.csv")
    //    val test_features = model.transform(tokenizer.transform(test_df))
    //    var predictions = logistic.transform(test_features)
    //    println("Predictions on test data")
    //    predictions.show()
    //
    //    println("Predictions on validation set")
    //    predictions = logistic.transform(testData)
    //    predictions.show()

    //    // Select (prediction, true label) and compute test error
    //    val evaluator = new MulticlassClassificationEvaluator()
    //      .setLabelCol("Sentiment")
    //      .setPredictionCol("prediction")
    //      .setMetricName("accuracy")
    //    val accuracy = evaluator.evaluate(predictions)
    //    println(s"Validation set accuracy = $accuracy")
  }
}