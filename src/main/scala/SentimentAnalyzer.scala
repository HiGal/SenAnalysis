import java.util.Calendar

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{LinearSVCModel, LogisticRegressionModel, MultilayerPerceptronClassificationModel, RandomForestClassificationModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SaveMode, SparkSession}
import org.apache.spark.streaming._

object SentimentAnalyzer {
  def main(args: Array[String]) {
    // Create Session
    val spark = SparkSession.builder
      .appName("Sentiment Analyzer")
      .getOrCreate

    // Load models
    val streamPipeline = PipelineModel.load("./pipeline")
    val logistic = LogisticRegressionModel.load("./logistic.model")
    val svm = LinearSVCModel.load("./svm.model")
    val perceptron = MultilayerPerceptronClassificationModel.load("./perceptron.model")
    val forest = RandomForestClassificationModel.load("./forest.model")
    // Create map for every model
    val models = Map("logistic" -> logistic, "svc"->svm, "perceptron"->perceptron, "random_forest"->forest)

    // Init stream reading
    val ssc = new StreamingContext(spark.sparkContext, Seconds(300))

    // Read stream line by line
    val lines = ssc.socketTextStream("10.90.138.32", 8989)
    val schema = new StructType()
      .add(StructField("SentimentText", StringType, true))

    val regex = "[,.:;'\"\\?\\-!\\(\\)]".r
    val stopwords = List("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now")
    val words = lines
      .map(line => "&(lt)?(gt)?(amp)?(quot)?;".r.replaceAllIn(line, ""))
      .map(line => "@[a-zA-Z0-9_]+".r.replaceAllIn(line, ""))
      .flatMap(line => line.split("\\W"))
      .map(word => regex.replaceAllIn(word.trim.toLowerCase, ""))
      .filter(word => !word.isEmpty)
      .filter(word => !stopwords.contains(word))
      .map(word => (word, 1))
      .reduceByKey(_ + _)
      .map(tuple => (tuple._2, tuple._1))
      .transform(rdd => rdd.sortByKey(false))
    words.saveAsTextFiles("topic/")

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

      for ((k,model) <- models){
        val predicitions = model.transform(features).
          select("prediction")
          .withColumn("Time", lit(s"$currentHour:$currentMinute"))
          .crossJoin(stream)
        //Then save the output
        predicitions.select("Time", "SentimentText", "prediction").withColumn("prediction", predicitions.col("prediction").cast(IntegerType))
          .write.mode(SaveMode.Append).csv("output/"+k)
      }
    })

    //Begin reading the stream
    ssc.start()
    ssc.awaitTermination()
  }
}