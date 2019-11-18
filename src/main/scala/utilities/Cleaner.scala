package utilities

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap, Params, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

trait HasRegExp extends Params {
  final val regularExp : StringArrayParam = new StringArrayParam(this, "inputCols", "input column names")

  /** @group getParam */
  final def getInputCols: Array[String] = $(regularExp)
}
class Cleaner(override val uid: String) extends Transformer with DefaultParamsWritable with HasRegExp {

  def this() = this(Identifiable.randomUID("RegExpCleaner"))
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def getOutputCol: String = getOrDefault(outputCol)
  def setRegularExpressions(value: Array[String]): this.type = set(regularExp, value)
  val inputCol = new Param[String](this, "inputCol", "input column")
  val outputCol = new Param[String](this, "outputCol", "output column")

  override def transform(dataset: Dataset[_]): DataFrame = {
    var column = dataset.col($(inputCol))
    for (regExp <- $(regularExp)){
      column = regexp_replace(column, regExp,"")
    }
    dataset.withColumn($(outputCol), column)
  }

  override def copy(extra: ParamMap): Cleaner = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = schema
}

object Cleaner extends DefaultParamsReadable[Cleaner] {
  override def load(path: String): Cleaner = super.load(path)
}
