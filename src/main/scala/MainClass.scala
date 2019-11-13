import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.log4j.Level
import model.word2vec
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.sql.SparkSession


object MainClass{
  def main(args: Array[String]){

    val spark = SparkSession.builder
      .master("local")
      .appName("Spark CSV Reader")
      .getOrCreate

    val df = spark.read
      .format("csv")
      .option("header", "true")
      .load("dataset/train.csv")

    val tokenizer = new RegexTokenizer()
      .setInputCol("SentimentText")
      .setOutputCol("Tokens")
      .setPattern("\\W+")
      .setGaps(true)

    val tmp = tokenizer.transform(df)

    val model = new word2vec().train(tmp,1)
    val result = model.transform(tmp)
  }
}