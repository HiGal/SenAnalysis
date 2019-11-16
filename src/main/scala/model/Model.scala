//package model
//import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
//import org.deeplearning4j.scalnet.layers.core.Dense
//import org.deeplearning4j.scalnet.layers.recurrent.{Bidirectional, LSTM}
//import org.deeplearning4j.scalnet.models.Sequential
//import org.nd4j.linalg.activations.Activation
//import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
//import org.deeplearning4j.optimize.listeners.ScoreIterationListener
//
//class Model {
//
//
//  def train(dataset: DataFrame, maxIter: Int): Sequential =
//  {
//    val seed: Int = 5
//
//    val spark = SparkSession.builder
//      .master("local")
//      .appName("Model training")
//      .getOrCreate
//
//    import spark.implicits._
//
//
//    case class df_ds(features:Vector[Double],label:Int)
//    val df = dataset.select("normedW2V", "Sentiment")
//    val ds = df.as[df_ds]
//
//    val model: Sequential = Sequential(rngSeed = seed)
//
//    model.add(Bidirectional(LSTM(128, 256, dropOut = 0.2)))
//    model.add(Dense(512, 256, activation = Activation.RELU))
//    model.add(Dense(2, 256, activation = Activation.SOFTMAX))
//
//    model.compile(LossFunction.NEGATIVELOGLIKELIHOOD)
//    model.fit(ds, maxIter, List(new ScoreIterationListener(100)))
//    model.evaluate(dataset).f1()
//
//  }
//}
