package model

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.example.utils.SimpleTokenizer._
import com.intel.analytics.bigdl.example.utils.{SimpleTokenizer, WordMeta}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, _}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.Source
import scala.language.existentials

/**
  * This example use a (pre-trained GloVe embedding) to convert word to vector,
  * and uses it to train a text classification model
  */
class TextClassifier(param: AbstractTextClassificationParams) extends Serializable {
  // defining global parameters
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  val gloveDir = s"${param.baseDir}/glove.6B/"
  val trainDataDir = s"${param.baseDir}/train.csv"
  val testDataDir = s"${param.baseDir}/test.csv"
  var classNum = -1

  /**
    * Load the pre-trained word2Vec
    *
    * @return A map from word to vector
    */
  def buildWord2Vec(word2Meta: Map[String, WordMeta]): Map[Float, Array[Float]] = {
    log.info("Indexing word vectors.")
    val preWord2Vec = MMap[Float, Array[Float]]()
    val filename = s"$gloveDir/glove.6B.200d.txt"
    for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      if (word2Meta.contains(word)) {
        val coefs = values.slice(1, values.length).map(_.toFloat)
        preWord2Vec.put(word2Meta(word).index.toFloat, coefs)
      }
    }
    log.info(s"Found ${preWord2Vec.size} word vectors.")
    preWord2Vec.toMap
  }

  /**
    * Load the pre-trained word2Vec
    *
    * @return A map from word to vector
    */
  def buildWord2VecWithIndex(word2Meta: Map[String, Int]): Map[Float, Array[Float]] = {
    log.info("Indexing word vectors.")
    val preWord2Vec = MMap[Float, Array[Float]]()
    val filename = s"$gloveDir/glove.6B.200d.txt"
    for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      if (word2Meta.contains(word)) {
        val coefs = values.slice(1, values.length).map(_.toFloat)
        preWord2Vec.put(word2Meta(word).toFloat, coefs)
      }
    }
    log.info(s"Found ${preWord2Vec.size} word vectors.")
    preWord2Vec.toMap
  }


  /**
    * Load the training data from the given baseDir
    *
    * @return An array of sample
    */
  private def loadRawData(train: Boolean): ArrayBuffer[(String, Float)] = {
    val texts = ArrayBuffer[String]()
    val labels = ArrayBuffer[Float]()
    val source = Source.fromFile(testDataDir, "ISO-8859-1")
    if (train) {
      val source = Source.fromFile(trainDataDir, "ISO-8859-1")
    }

    var text = try source.getLines().toList finally source.close()
    text = text.slice(1,text.length)

    text.foreach { line =>
      val arr = line.split(",")
      texts.append(arr(2))
      labels.append(arr(1).toFloat)
    }


    this.classNum = labels.toSet.size
    log.info(s"Found ${texts.length} texts.")
    log.info(s"Found $classNum classes")
    texts.zip(labels)

  }

  /**
    * Go through the whole data set to gather some meta info for the tokens.
    * Tokens would be discarded if the frequency ranking is less then maxWordsNum
    */
  def analyzeTexts(dataRdd: RDD[(String, Float)])
  : (Map[String, WordMeta], Map[Float, Array[Float]]) = {
    val frequencies = dataRdd.flatMap { case (text: String, label: Float) =>
      SimpleTokenizer.toTokens(text)
    }.map(word => (word, 1)).reduceByKey(_ + _)
      .sortBy(-_._2).collect().slice(10, param.maxWordsNum)

    val indexes = Range(1, frequencies.length)
    val word2Meta = frequencies.zip(indexes).map { item =>
      (item._1._1, WordMeta(item._1._2, item._2))
    }.toMap
    (word2Meta, buildWord2Vec(word2Meta))
  }

  /**
    * Create train and val RDDs from input
    */
  def getData(sc: SparkContext): (Array[RDD[(Array[Array[Float]], Float)]],
    Map[String, WordMeta],
    Map[Float, Array[Float]]) = {

    val sequenceLen = param.maxSequenceLength
    val embeddingDim = param.embeddingDim
    val trainingSplit = param.trainingSplit
    // For large dataset, you might want to get such RDD[(String, Float)] from HDFS
    val dataRdd = sc.parallelize(loadRawData(true), param.partitionNum)
    val (word2Meta, word2Vec) = analyzeTexts(dataRdd)
    val word2MetaBC = sc.broadcast(word2Meta)
    val word2VecBC = sc.broadcast(word2Vec)
    val vectorizedRdd = dataRdd
      .map { case (text, label) => (toTokens(text, word2MetaBC.value), label) }
      .map { case (tokens, label) => (shaping(tokens, sequenceLen), label) }
      .map { case (tokens, label) => (vectorization(
        tokens, embeddingDim, word2VecBC.value), label)
      }

    (vectorizedRdd.randomSplit(
      Array(trainingSplit, 1 - trainingSplit)), word2Meta, word2Vec)

  }

  /**
    * Return a text classification model with the specific num of
    * class
    */
  def buildModel(classNum: Int): Sequential[Float] = {
    val model = Sequential[Float]()

    model.add(TemporalConvolution(param.embeddingDim, 256, 5))
      .add(ReLU())
      .add(TemporalMaxPooling(param.maxSequenceLength - 5 + 1))
      .add(Squeeze(2))
      .add(Linear(256, 128))
      .add(Dropout(0.2))
      .add(ReLU())
      .add(Linear(128, classNum))
      .add(LogSoftMax())
    model
  }


  /**
    * Start to train the text classification model
    */
  def train(): Unit = {
        val conf = Engine.createSparkConf()
          .setAppName("Text classification")
          .set("spark.task.maxFailures", "1")

        val sc = new SparkContext(conf)
        Engine.init
    val sequenceLen = param.maxSequenceLength
    val embeddingDim = param.embeddingDim
    val trainingSplit = param.trainingSplit

    // For large dataset, you might want to get such RDD[(String, Float)] from HDFS
    val dataRdd = sc.parallelize(loadRawData(true), param.partitionNum)

    val (word2Meta, word2Vec) = analyzeTexts(dataRdd)
    val word2MetaBC = sc.broadcast(word2Meta)
    val word2VecBC = sc.broadcast(word2Vec)
    val vectorizedRdd = dataRdd
      .map { case (text, label) => (toTokens(text, word2MetaBC.value), label) }
      .map { case (tokens, label) => (shaping(tokens, sequenceLen), label) }
      .map { case (tokens, label) => (vectorization(
        tokens, embeddingDim, word2VecBC.value), label)
      }
    val sampleRDD = vectorizedRdd.map { case (input: Array[Array[Float]], label: Float) =>
      Sample(
        featureTensor = Tensor(input.flatten, Array(sequenceLen, embeddingDim)),
        label = label)
    }

    val Array(trainingRDD, valRDD) = sampleRDD.randomSplit(
      Array(trainingSplit, 1 - trainingSplit))

    // Create optimizer
    val optimizer = Optimizer(
      model = buildModel(classNum),
      sampleRDD = trainingRDD,
      criterion = new ClassNLLCriterion[Float](),
      batchSize = param.batchSize
    )

    // Begin training
    optimizer
      .setOptimMethod(new Adagrad(learningRate = param.learningRate,
        learningRateDecay = 0.001))
      .setValidation(Trigger.everyEpoch, valRDD, Array(new Top1Accuracy[Float]), param.batchSize)
      .setEndWhen(Trigger.maxEpoch(20))
      .optimize()

    sc.stop()
  }

  /**
    * Train the text classification model with train and val RDDs
    */
  def trainFromData(sc: SparkContext, rdds: Array[RDD[(Array[Array[Float]], Float)]])
  : Module[Float] = {

    // create rdd from input directory
    val trainingRDD = rdds(0).map { case (input: Array[Array[Float]], label: Float) =>
      Sample(
        featureTensor = Tensor(input.flatten, Array(param.maxSequenceLength, param.embeddingDim))
          .transpose(1, 2).contiguous(),
        label = label)
    }

    val valRDD = rdds(1).map { case (input: Array[Array[Float]], label: Float) =>
      Sample(
        featureTensor = Tensor(input.flatten, Array(param.maxSequenceLength, param.embeddingDim))
          .transpose(1, 2).contiguous(),
        label = label)
    }

    // train
    val optimizer = Optimizer(
      model = buildModel(classNum),
      sampleRDD = trainingRDD,
      criterion = new ClassNLLCriterion[Float](),
      batchSize = param.batchSize
    )


    optimizer
      .setOptimMethod(new Adagrad(learningRate = param.learningRate, learningRateDecay = 0.0002))
      .setValidation(Trigger.everyEpoch, valRDD, Array(new Top1Accuracy[Float]), param.batchSize)
      .setEndWhen(Trigger.maxEpoch(1))
      .optimize()
  }
}


abstract class AbstractTextClassificationParams extends Serializable {
  def baseDir: String = "./"

  def maxSequenceLength: Int = 1000

  def maxWordsNum: Int = 20000

  def trainingSplit: Double = 0.8

  def batchSize: Int = 128

  def embeddingDim: Int = 100

  def partitionNum: Int = 4

  def learningRate: Double = 0.01
}


/**
  * @param baseDir           The root directory which containing the training and embedding data
  * @param maxSequenceLength number of the tokens
  * @param maxWordsNum       maximum word to be included
  * @param trainingSplit     percentage of the training data
  * @param batchSize         size of the mini-batch
  * @param learningRate      learning rate
  * @param embeddingDim      size of the embedding vector
  */
case class TextClassificationParams(override val baseDir: String = "./",
                                    override val maxSequenceLength: Int = 500,
                                    override val maxWordsNum: Int = 5000,
                                    override val trainingSplit: Double = 0.8,
                                    override val batchSize: Int = 128,
                                    override val embeddingDim: Int = 200,
                                    override val learningRate: Double = 0.01,
                                    override val partitionNum: Int = 4)
  extends AbstractTextClassificationParams
