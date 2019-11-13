import model.word2vec
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.log4j.Level
import org.apache.spark.sql.SparkSession

object MainClass{
  def main(args: Array[String]){
    val spark = SparkSession.builder
      .master("local[4]")
      .appName("Main")
      .getOrCreate
    val df = spark.read
      .format("csv")
      .option("header", "true") //first line in file has headers
      .load("dataset/train.csv")
    val model = new word2vec()
    model.train(df)
    // Configure Twitter credentials using twitter.txt
//    setupTwitter()
//    val ssc = new StreamingContext("local[*]", "PrintTweets", Seconds(1))
//    setupLogging()
//
//    // Create a DStream from Twitter using our streaming context
//    val tweets = TwitterUtils.createStream(ssc, None)
//
//    // Now extract the text of each status update into RDD's using map()
//    val statuses = tweets.map(status => status.getText())
//    statuses.print()
//
//    ssc.start()
//    ssc.awaitTermination()

  }

}