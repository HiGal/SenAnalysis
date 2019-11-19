import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, _}
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T}
import model.{TextClassificationParams, TextClassifier}
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import org.apache.spark
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.slf4j.{Logger, LoggerFactory}
import scopt.OptionParser

import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.language.existentials

object TextClassifierTrain {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  LoggerFilter.redirectSparkInfoLogs()
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Levle4j.INFO)



  def main(args: Array[String]): Unit = {

    // Argument Parser
    val localParser = new OptionParser[TextClassificationParams]("BigDL Example") {
      opt[String]('b', "baseDir")
        .required()
        .text("Base dir containing the training and word2Vec data")
        .action((x, c) => c.copy(baseDir = x))
      opt[String]('p', "partitionNum")
        .text("you may want to tune the partitionNum if run into spark mode")
        .action((x, c) => c.copy(partitionNum = x.toInt))
      opt[String]('s', "maxSequenceLength")
        .text("maxSequenceLength")
        .action((x, c) => c.copy(maxSequenceLength = x.toInt))
      opt[String]('w', "maxWordsNum")
        .text("maxWordsNum")
        .action((x, c) => c.copy(maxWordsNum = x.toInt))
      opt[String]('l', "trainingSplit")
        .text("trainingSplit")
        .action((x, c) => c.copy(trainingSplit = x.toDouble))
      opt[String]('z', "batchSize")
        .text("batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('l', "learningRate")
        .text("learningRate")
        .action((x, c) => c.copy(learningRate = x))
    }


    // Train model
    localParser.parse(args, TextClassificationParams()).map { param =>
      log.info(s"Current parameters: $param")
      val textClassification = new TextClassifier(param)
      textClassification.train()
    }
  }
}
