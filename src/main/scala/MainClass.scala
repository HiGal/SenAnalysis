import model.word2vec
import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.log4j.Logger
import org.apache.spark.streaming.StreamingContext._
import org.apache.log4j.Level
import org.apache.spark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.{Column, Row, SaveMode, SparkSession}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{Normalizer, RegexTokenizer, StopWordsRemover, VectorAssembler}
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._
import java.util.Calendar

object MainClass {
  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .master("local")
      .appName("Stream Analyzer")
      .getOrCreate

    // Load models
    val streamPipeline = PipelineModel.load("./pipeline")
    val logistic = LogisticRegressionModel.load("./logistic.model")

    val models = Map("logistic" -> logistic) // Add More models later
    val ssc = new StreamingContext(spark.sparkContext, Seconds(60))

    // Words reading in stream
    val lines = ssc.socketTextStream("10.90.138.32", 8989)
    val schema = new StructType()
      .add(StructField("SentimentText", StringType, true))
    lines.foreachRDD(rdd => {
      val dT = Calendar.getInstance()
      val currentMinute = dT.get(Calendar.MINUTE)
      val currentHour = dT.get(Calendar.HOUR_OF_DAY)
      val stream = spark.createDataFrame(rdd.map(attributes => Row(attributes)), schema).withColumn("id", monotonicallyIncreasingId)
      val features = streamPipeline.transform(stream)
      for ((k,v) <- models){
        val predicitions = v.transform(features).
          select("prediction")
          .withColumn("Time", lit(s"$currentHour:$currentMinute"))
          .crossJoin(stream)
        predicitions.select("Time", "SentimentText", "prediction")
          .write.mode(SaveMode.Append).csv("output/"+k+".csv")
      }
    })
    //TODO: write count words
    ssc.start()
    ssc.awaitTermination()
  }
}