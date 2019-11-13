package model

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer, Word2Vec, Word2VecModel}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

class word2vec {
  def train(dataset: DataFrame, maxIter: Int): Word2VecModel = {


    val model = new Word2Vec()
      .setMaxIter(maxIter)
      .setVectorSize(128)
      .setInputCol("Tokens")
      .setOutputCol("result")


    val w2vmodel = model.fit(dataset)

    w2vmodel
  }

}
