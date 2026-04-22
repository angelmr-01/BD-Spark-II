package es.upm.bd

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, StringIndexer, PCA}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.classification.{RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, DecisionTreeClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.Pipeline

object PracticaSparkML {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Analisis_Consumo_Electrico_Final")
      // .master("local[*]")
      .getOrCreate()
      
    spark.sparkContext.setLogLevel("WARN")

    println("=====================================================")
    println(" INICIANDO PIPELINE DE MACHINE LEARNING ")
    println("=====================================================")

    // ==============================================================================
    // FASE 1: CARGA Y LIMPIEZA DE DATOS (DataFrames)
    // ==============================================================================
    val schemaDefinition = StructType(
      Array(
        StructField("IDENTIFICADOR", StringType, true),
        StructField("ANOMES", IntegerType, true),
        StructField("CNAE", StringType, true),
        StructField("PRODUCTO", StringType, true),
        StructField("MERCADO", StringType, true)
      ) ++ 
      (1 to 25).map(i => StructField(s"ACTIVA_H$i", DoubleType, true)) ++
      (1 to 25).map(i => StructField(s"REACTIVA_H$i", DoubleType, true))
    )

    val dfBruto = spark.read
      .option("header", "false")
      .option("ignoreLeadingWhiteSpace", "true")
      .option("delimiter",",")
      .schema(schemaDefinition)
      .csv("file:///opt/spark/data/endesaAgregada")

    val columnasABorrar = Seq("ACTIVA_H25") ++ (1 to 25).map(i => s"REACTIVA_H$i")
    val dfFiltrado = dfBruto.drop(columnasABorrar: _*).na.drop()


    // ==============================================================================
    // FASE 2: PREPROCESAMIENTO COMUN
    // ==============================================================================
    val columnasActiva = (1 to 24).map(i => s"ACTIVA_H$i").toArray
    val assembler = new VectorAssembler()
      .setInputCols(columnasActiva)
      .setOutputCol("featuresO")

    val scaler = new StandardScaler()
      .setInputCol("featuresO")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)

    val labelIndexer = new StringIndexer()
      .setInputCol("MERCADO")
      .setOutputCol("label")


    // ==============================================================================
    // FASE 3: APRENDIZAJE NO SUPERVISADO (K-Means y PCA)
    // ==============================================================================
    println("\n--> FASE 3: NO SUPERVISADO (Clusters y Extracción Principal)...")
    val evaluadorCluster = new ClusteringEvaluator()

    for (k <- 2 to 4) {
      val kmeansTest = new KMeans().setK(k).setSeed(1L).setFeaturesCol("features")
      val pipelineKMeans = new Pipeline().setStages(Array(assembler, scaler, kmeansTest))
      val preds = pipelineKMeans.fit(dfFiltrado).transform(dfFiltrado)
      val silhouette = evaluadorCluster.evaluate(preds)
      println(s"  [*] Evaluando K=$k K-Means -> Silhouette: $silhouette")
    }

    println("  [*] Test Final con K-Means (K=2) y compresión geométrica (PCA k=2):")
    val kmeansFinal = new KMeans().setK(2).setSeed(1L).setFeaturesCol("features")
    val dfClusters = new Pipeline().setStages(Array(assembler, scaler, kmeansFinal)).fit(dfFiltrado).transform(dfFiltrado)
    dfClusters.groupBy("prediction").count().show()
    
    val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(2)
    val dfPCA = pca.fit(dfClusters).transform(dfClusters)
    println("  [+] Compresión PCA completada.")


    // ==============================================================================
    // FASE 4: APRENDIZAJE SUPERVISADO (Clasificacion Binaria con ROC)
    // ==============================================================================
    println("\n--> FASE 4: SUPERVISADO (Mediciones ROC)...")
    
    val Array(trainingData, testData) = dfFiltrado.randomSplit(Array(0.7, 0.3), seed = 1234L)
    
    val evaluadorROC = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    println("  [*] Baseline: Regresión Logística")
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10)
    val pipelineLR = new Pipeline().setStages(Array(assembler, scaler, labelIndexer, lr))
    val lrROC = evaluadorROC.evaluate(pipelineLR.fit(trainingData).transform(testData))
    println(s"  [-] Área bajo la Curva ROC (LogisticRegression): $lrROC")

    println("  [*] Entrenando CrossValidator con Random Forest (Esperar...)")
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val pipelineRF = new Pipeline().setStages(Array(assembler, scaler, labelIndexer, rf))

    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(10, 20))
      //.addGrid(rf.maxDepth, Array(5, 7)) -> Optimizar tiempo local
      .build()

    val crossval = new CrossValidator()
      .setEstimator(pipelineRF)
      .setEvaluator(evaluadorROC)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
      .setSeed(1234L)

    val cvModel = crossval.fit(trainingData)
    val cvPreds = cvModel.transform(testData)
    val cvROC = evaluadorROC.evaluate(cvPreds)
    println(s"  [+] Área bajo la Curva ROC (RF + K-Folds): $cvROC")

    println("\n  [*] Extrayendo Árbol de Explicabilidad (DecisionTree)...")
    val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").setMaxDepth(3)
    val pipelineDT = new Pipeline().setStages(Array(assembler, scaler, labelIndexer, dt))
    val pDTFit = pipelineDT.fit(trainingData)
    val dtROC = evaluadorROC.evaluate(pDTFit.transform(testData))
    
    val arbolAislado = pDTFit.stages(3).asInstanceOf[DecisionTreeClassificationModel]
    println(s"  [+] Área bajo la Curva ROC (Árbol Simple): $dtROC")
    println(" \n====== REGLAS DE NEGOCIO OBTENIDAS ======")
    println(arbolAislado.toDebugString)

    println("\n  [*] Entrenando Gradient-Boosted Trees (Top Rendimiento Tabular)...")
    val gbt = new GBTClassifier().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setMaxDepth(5)
    val pipelineGBT = new Pipeline().setStages(Array(assembler, scaler, labelIndexer, gbt))
    val gbtROC = evaluadorROC.evaluate(pipelineGBT.fit(trainingData).transform(testData))
    println(s"  [+] Área bajo la Curva ROC (Gradient-Boosted Trees): $gbtROC")

    println("=====================================================")
    println(" PIPELINE FINALIZADO CON ÉXITO ")
    println("=====================================================")

    spark.stop()
  }
}
