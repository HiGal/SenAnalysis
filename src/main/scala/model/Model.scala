package model
import org.deeplearning4j.scalnet.layers.core.Dense
import org.deeplearning4j.scalnet.layers.embeddings.EmbeddingLayer
import org.deeplearning4j.scalnet.layers.recurrent.{Bidirectional, LSTM}
import org.deeplearning4j.scalnet.models.Sequential
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.apache.spark.mllib.feature.Word2Vec
import org.nd4j.weightinit.WeightInit
import org.nd4j.weightinit.impl.XavierInitScheme
import org.apache.spark.sql.SparkSession

class Model {

  val seed: Int = 5



  val vec = new Word2Vec()

  val model: Sequential = Sequential(rngSeed = seed)

  model.add(Bidirectional(LSTM(100, 256, dropOut = 0.2)))
  model.add(Dense(512, 256,activation = Activation.RELU))
  model.add(Dense(2, 256, activation = Activation.SOFTMAX))

  model.compile(LossFunction.NEGATIVELOGLIKELIHOOD)
}
