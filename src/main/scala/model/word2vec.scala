package model

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer, Word2Vec, Word2VecModel}
import org.apache.spark.sql.{SQLContext, SparkSession, DataFrame}

class word2vec {

  val model: Word2Vec = new Word2Vec().setMaxIter(5).setVectorSize(128).setInputCol("Tokens").setOutputCol("Result")
  var trained_model: Word2VecModel = _

  def train(df: DataFrame): Word2VecModel = {
    val tokenizer = new RegexTokenizer()
      .setInputCol("SentimentText")
      .setOutputCol("Tokens")
      .setPattern("\\W+")
      .setGaps(true)
    val tmp = tokenizer.transform(df)
    tmp.show(10)
    trained_model = model.fit(tmp)
    val result = trained_model.transform(tmp)
    result.select("result").take(3).foreach(println)
    trained_model
  }

  def save_model(path:String): Unit = {
    trained_model.save(path)
  }

  def load_model(path:String): Word2VecModel ={
    trained_model = Word2VecModel.load(path)
    trained_model
  }
}
