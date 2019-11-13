name := "BD-A2"

version := "0.1"

scalaVersion := "2.11.8"
val sparkVersion = "2.4.4"

resolvers += "snapshots-repo" at "https://oss.sonatype.org/content/repositories/snapshots"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.deeplearning4j" % "scalnet_2.11" % "0.9.2-SNAPSHOT",
  "org.apache.spark" %% "spark-mllib"%"2.4.0"
)
