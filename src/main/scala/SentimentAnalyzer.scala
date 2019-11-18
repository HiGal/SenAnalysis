import java.util.Calendar

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SaveMode, SparkSession}
import org.apache.spark.streaming._

object SentimentAnalyzer {
  def main(args: Array[String]) {

    val updateFunc = (values: Seq[Int], state: Option[Int]) => {
      val currentCount = values.foldLeft(0)(_ + _)

      val previousCount = state.getOrElse(0)

      Some(currentCount + previousCount)
    }

    // Create Session
    val spark = SparkSession.builder
      .master("local")
      .appName("Sentiment Analyzer")
      .getOrCreate

    // Load models
    val streamPipeline = PipelineModel.load("./pipeline")
    val logistic = LogisticRegressionModel.load("./logistic.model")

    // Create map for every model
    val models = Map("logistic" -> logistic)

    // Init stream reading
    val ssc = new StreamingContext(spark.sparkContext, Seconds(60))

    // Read stream line by line
    val lines = ssc.socketTextStream("10.90.138.32", 8989)
    val schema = new StructType()
      .add(StructField("SentimentText", StringType, true))
    lines.foreachRDD(rdd => {
      // Get current time
      val dT = Calendar.getInstance()
      val currentMinute = dT.get(Calendar.MINUTE)
      val currentHour = dT.get(Calendar.HOUR_OF_DAY)

      // Create DataFrame from input
      val stream = spark.createDataFrame(rdd.map(attributes => Row(attributes)), schema).withColumn("id", monotonicallyIncreasingId)

      // Apply pipeline transformation before using models
      val features = streamPipeline.transform(stream)
      // On each feature use pretrained model to predict result
      val word_count = features.select("filteredPhrases").rdd.map(r => r(0)).collect()
      println(word_count)

      for ((k,model) <- models){
        val predicitions = model.transform(features).
          select("prediction")
          .withColumn("Time", lit(s"$currentHour:$currentMinute"))
          .crossJoin(stream)
        //Then save the output
        predicitions.select("Time", "SentimentText", "prediction")
          .coalesce(1).write.mode(SaveMode.Append).csv("output/"+k)
      }
    })

    //Begin reading the stream
    ssc.start()
    ssc.awaitTermination()
  }
}