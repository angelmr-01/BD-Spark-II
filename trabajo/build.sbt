name := "Entrega3-BigData"

version := "1.0"

scalaVersion := "2.12.18" // Versión compatible con Spark 3.5

val sparkVersion = "3.5.5" // La versión que usamos en el docker-compose

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided"
)
