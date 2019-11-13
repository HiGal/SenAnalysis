package model

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer, Word2Vec, Word2VecModel}
import org.apache.spark.sql.{SQLContext, SparkSession, DataFrame}

class word2vec {
  var trained_model: Word2VecModel = _

  def train(dataset: DataFrame, maxIter: Int): Word2VecModel = {


    val model = new Word2Vec()
      .setMaxIter(maxIter)
      .setVectorSize(128)
      .setInputCol("Tokens")
      .setOutputCol("result")
    trained_model = model.fit(dataset)
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
