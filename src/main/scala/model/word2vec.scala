package model


import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, SparkSession}

object word2vec {
  def main(args: Array[String]): Unit = {
//    val conf = new SparkConf().setAppName("Word2vec")
//    val sc = new SparkContext(conf)
    val spark = SparkSession.builder
      .master("local")
      .appName("Spark CSV Reader")
      .getOrCreate
    val model = new Word2Vec()
    model.setVectorSize(128)
    model.setInputCol("SentimentText")
    val df = spark.read
      .format("csv")
      .option("header", "true") //first line in file has headers
      .option("mode", "DROPMALFORMED")
      .load("dataset/train.csv")
    import spark.implicits._
    val tmp = df.select("SentimentText").map(line => line.getString(0).split(" ")).toDF("SentimentText")
    model.fit(tmp)
  }

}
