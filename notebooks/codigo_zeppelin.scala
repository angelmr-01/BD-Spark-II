// =======================================================================
// CELDA 1: IMPORTS Y CONFIGURACIÓN INICIAL
// =======================================================================
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, StringIndexer, PCA}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.classification.{RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, DecisionTreeClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.Pipeline

spark.sparkContext.setLogLevel("WARN")

// =======================================================================
// CELDA 2: CARGA DE DATOS (DataFrame)
// =======================================================================
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

// Eliminamos H25 y las reactivas
val columnasABorrar = Seq("ACTIVA_H25") ++ (1 to 25).map(i => s"REACTIVA_H$i")
val dfFiltrado = dfBruto.drop(columnasABorrar: _*).na.drop()

println(s"Total registros tras limpieza inicial: ${dfFiltrado.count()}")

// =======================================================================
// CELDA 3: DECLARACIÓN DE FEATURES COMUNES PARA PIPELINES
// =======================================================================
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


// =======================================================================
// CELDA 4: APRENDIZAJE NO SUPERVISADO (K-MEANS)
// =======================================================================
val evaluadorCluster = new ClusteringEvaluator()

println(">> Evaluando Iteraciones para K-Means...")
for (k <- 2 to 5) {
  val kmeans = new KMeans().setK(k).setSeed(1L).setFeaturesCol("features")
  val pipelineKMeans = new Pipeline().setStages(Array(assembler, scaler, kmeans))
  val preds = pipelineKMeans.fit(dfFiltrado).transform(dfFiltrado)
  val silhouette = evaluadorCluster.evaluate(preds)
  println(s"  [*] Silhouette K-Means (K=$k): $silhouette")
}

// Entrenamos con K=2 porque es el k que mejores resultados ha obtenido
val kmeansFinal = new KMeans().setK(2).setSeed(1L).setFeaturesCol("features")
val modeloKMeansFinal = new Pipeline().setStages(Array(assembler, scaler, kmeansFinal)).fit(dfFiltrado)
val dfClusters = modeloKMeansFinal.transform(dfFiltrado)

println(">> Conteo K-Means (K=2):")
dfClusters.groupBy("prediction").count().show()


// =======================================================================
// CELDA 4.2: EXPLICABILIDAD Y SEGUNDO MÉTODO NO SUPERVISADO (PCA)
// =======================================================================
println(">> Evaluando Análisis de Componentes Principales (PCA k=2)...")
val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(2)
val dfPCA = pca.fit(dfClusters).transform(dfClusters)

// Extraemos los vectores para poder graficar X e Y planos en Zeppelin
val extractPC1 = udf((v: Vector) => v(0))
val extractPC2 = udf((v: Vector) => v(1))

val plotScatterPCA = dfPCA.select(
  extractPC1(col("pcaFeatures")).alias("PC1"),
  extractPC2(col("pcaFeatures")).alias("PC2"),
  col("prediction").cast(StringType).alias("Cluster") // Convertido a String para forzar categoría
)

// Gráfica del perfil medio de horas (Líneas)
val dfPromedios = dfClusters.groupBy("prediction").agg(
    avg("ACTIVA_H1").alias("H1"), avg("ACTIVA_H2").alias("H2"), avg("ACTIVA_H3").alias("H3"),
    avg("ACTIVA_H4").alias("H4"), avg("ACTIVA_H5").alias("H5"), avg("ACTIVA_H6").alias("H6"),
    avg("ACTIVA_H7").alias("H7"), avg("ACTIVA_H8").alias("H8"), avg("ACTIVA_H9").alias("H9"),
    avg("ACTIVA_H10").alias("H10"),avg("ACTIVA_H11").alias("H11"),avg("ACTIVA_H12").alias("H12"),
    avg("ACTIVA_H13").alias("H13"),avg("ACTIVA_H14").alias("H14"),avg("ACTIVA_H15").alias("H15"),
    avg("ACTIVA_H16").alias("H16"),avg("ACTIVA_H17").alias("H17"),avg("ACTIVA_H18").alias("H18"),
    avg("ACTIVA_H19").alias("H19"),avg("ACTIVA_H20").alias("H20"),avg("ACTIVA_H21").alias("H21"),
    avg("ACTIVA_H22").alias("H22"),avg("ACTIVA_H23").alias("H23"),avg("ACTIVA_H24").alias("H24")
)

val plotLineas = dfPromedios.selectExpr(
  "prediction as Cluster",
  "stack(24, 1, H1, 2, H2, 3, H3, 4, H4, 5, H5, 6, H6, 7, H7, 8, H8, 9, H9, 10, H10, 11, H11, 12, H12, 13, H13, 14, H14, 15, H15, 16, H16, 17, H17, 18, H18, 19, H19, 20, H20, 21, H21, 22, H22, 23, H23, 24, H24) as (Hora, ConsumoMedio)"
).withColumn("Cluster", col("Cluster").cast(StringType))

println(">> INSTRUCCIONES GRÁFICAS PARA ZEPPELIN:")
println("   A) PERFIL MEDIO (Curva): Crear celda con 'z.show(plotLineas)'. Elegir Line Chart.")
println("      Configurar -> Keys: Hora | Groups: Cluster | Values: ConsumoMedio(AVG)")
println("   B) MAPA CLUSTERS 2D: Crear celda con 'z.show(plotScatterPCA)'. Elegir Scatter Chart.")
println("      Configurar -> xAxis: PC1 | yAxis: PC2 | group: Cluster")


// =======================================================================
// CELDA 5: APRENDIZAJE SUPERVISADO (Clasif Binaria. Pipeline ROC)
// =======================================================================
val Array(trainingData, testData) = dfFiltrado.randomSplit(Array(0.7, 0.3), seed = 1234L)

// Evaluamos con ROC en lugar de Accuracy para compensar desbalanceos
val evaluadorROC = new BinaryClassificationEvaluator()
  .setLabelCol("label")
  .setRawPredictionCol("rawPrediction")
  .setMetricName("areaUnderROC")

// ---> 5.1 Baseline: Regresión Logística (Línea Base Lineal)
val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10)
val pipelineLR = new Pipeline().setStages(Array(assembler, scaler, labelIndexer, lr))
val lrROC = evaluadorROC.evaluate(pipelineLR.fit(trainingData).transform(testData))
println(s">> Área Bajo la Curva ROC (Logística Baseline): $lrROC")

// ---> 5.2 Ensamblado: Random Forest con K-Folds Cross Validation
println(">> Configurando K-Folds Cross Validation para Random Forest...")
val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features")
val pipelineRF = new Pipeline().setStages(Array(assembler, scaler, labelIndexer, rf))

val paramGrid = new ParamGridBuilder()
  .addGrid(rf.numTrees, Array(10, 20))
  //.addGrid(rf.maxDepth, Array(5, 7))
  .build()

val crossval = new CrossValidator()
  .setEstimator(pipelineRF)
  .setEvaluator(evaluadorROC)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)
  .setSeed(1234L)

val cvModel = crossval.fit(trainingData)
val cvPredicciones = cvModel.transform(testData)
val cvROC = evaluadorROC.evaluate(cvPredicciones)
println(s">> Área Bajo la Curva ROC (RF con K-Folds): $cvROC")


// ---> 5.3 Árbol de Decisión (Reglas de Negocio)
println("\n>> Entrenando un Árbol de Decisión Simple para extraer Reglas de Clasificación...")
val dt = new DecisionTreeClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setMaxDepth(3)

val pipelineDT = new Pipeline().setStages(Array(assembler, scaler, labelIndexer, dt))
val modeloEntrenadoDT = pipelineDT.fit(trainingData)
val dtROC = evaluadorROC.evaluate(modeloEntrenadoDT.transform(testData))

// Extraemos el propio modelo de Árbol (es el último paso [Stage 3] en el pipeline Array)
val arbolAislado = modeloEntrenadoDT.stages(3).asInstanceOf[DecisionTreeClassificationModel]
println(s">> Área Bajo la Curva ROC (Árbol Simple): $dtROC")
println("\n====== REGLAS DE DECISIÓN OBTENIDAS ======")
println(arbolAislado.toDebugString)

// ---> 5.4 Búsqueda del Límite Matemático: Gradient-Boosted Trees (GBT)
println("\n>> Entrenando Gradient-Boosted Trees (GBT) para exprimir el rendimiento analítico...")
val gbt = new GBTClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setMaxIter(10)
  .setMaxDepth(5)

val pipelineGBT = new Pipeline().setStages(Array(assembler, scaler, labelIndexer, gbt))
val gbtROC = evaluadorROC.evaluate(pipelineGBT.fit(trainingData).transform(testData))
println(s">> Área Bajo la Curva ROC (Gradient-Boosted Trees): $gbtROC")
