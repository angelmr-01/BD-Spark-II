package es.upm.bd

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object PracticaSpark {

  // DEFINICIÓN DE CLASE PARA API TIPADA
  case class ConsumoElectrico(
    IDENTIFICADOR: String, ANOMES: Int, CNAE: String, PRODUCTO: String, MERCADO: String,
    ACTIVA_H1: Double, ACTIVA_H2: Double, ACTIVA_H3: Double, ACTIVA_H4: Double, ACTIVA_H5: Double,
    ACTIVA_H6: Double, ACTIVA_H7: Double, ACTIVA_H8: Double, ACTIVA_H9: Double, ACTIVA_H10: Double,
    ACTIVA_H11: Double, ACTIVA_H12: Double, ACTIVA_H13: Double, ACTIVA_H14: Double, ACTIVA_H15: Double,
    ACTIVA_H16: Double, ACTIVA_H17: Double, ACTIVA_H18: Double, ACTIVA_H19: Double, ACTIVA_H20: Double,
    ACTIVA_H21: Double, ACTIVA_H22: Double, ACTIVA_H23: Double, ACTIVA_H24: Double
  )

  def main(args: Array[String]): Unit = {

    // 0. ARRANQUE DEL CLÚSTER
    val spark = SparkSession.builder()
      .appName("Despliegue_Consumo_Electrico_Entrega3")
      // .master("local[*]") // Descomentar solo si se prueba dentro de IntelliJ localmente
      .getOrCreate()
      
    import spark.implicits._

    spark.sparkContext.setLogLevel("WARN")

    println("=====================================================")
    println(" INICIANDO PIPELINE DE DATOS")
    println("=====================================================")

    // ==============================================================================
    // BLOQUE 1: RDD
    // ==============================================================================
    println("\n--> EJECUTANDO BLOQUE 1 (RDD)...")
    val rddCrudo = spark.sparkContext.textFile("file:///opt/spark/data/endesaAgregada")
    
    val rddSeleccionado = rddCrudo.map(linea => {
      val cols = linea.split(",")
      val identificador = cols(0).trim
      val anomes = cols(1).trim
      val cnae = cols(2).trim
      val producto = cols(3).trim
      val mercado = cols(4).trim
      val consumosArray = cols.slice(5, 29).map(_.trim.toDouble)
      (identificador, anomes, cnae, producto, mercado, consumosArray)
    })

    val rddLimpio = rddSeleccionado.filter(caso => caso._3.nonEmpty && caso._4.nonEmpty && caso._6.forall(_ >= 0))

    val rddProcesado = rddLimpio.map(caso => {
      val (id, an, cn, pr, me, arr) = caso
      val totalActiva = arr.sum
      (id, an, cn, pr, me, totalActiva)
    })

    val rddTotalPorMercado = rddProcesado
      .map(x => (x._5, x._6))
      .reduceByKey(_ + _)
    // Convertimos a DF sólo para mostrar un snippet tabular limpio en consola
    rddTotalPorMercado.toDF("Tipo_Mercado", "Suma_Activa").show(5)


    // ==============================================================================
    // BLOQUE 2: DATAFRAMES (API NO TIPADA)
    // ==============================================================================
    println("\n--> EJECUTANDO BLOQUE 2 (DATAFRAMES)...")
    val schemaDefinition = StructType(
        (Array(
          StructField("IDENTIFICADOR", StringType, true),
          StructField("ANOMES", IntegerType, true),
          StructField("CNAE", StringType, true),
          StructField("PRODUCTO", StringType, true),
          StructField("MERCADO", StringType, true)
        ) ++ 
        (1 to 25).map(i => StructField(s"ACTIVA_H$i", DoubleType, true)) ++
        (1 to 25).map(i => StructField(s"REACTIVA_H$i", DoubleType, true))).toArray
    )

    val dfBruto = spark.read
        .option("header", "false")
        .option("ignoreLeadingWhiteSpace", "true")
        .option("delimiter",",")
        .schema(schemaDefinition)
        .csv("file:///opt/spark/data/endesaAgregada")

    val columnasABorrar = Seq("ACTIVA_H25") ++ (1 to 25).map(i => s"REACTIVA_H$i")
    val dfSeleccionado = dfBruto.drop(columnasABorrar: _*)
    
    val condicionPositivos = (1 to 24).map(i => col(s"ACTIVA_H$i") >= 0).reduce(_ && _)
    val dfLimpio = dfSeleccionado.na.drop(Seq("IDENTIFICADOR", "CNAE", "MERCADO")).filter(condicionPositivos)

    val dfFiltroAnomes = dfLimpio.filter(col("ANOMES") === 201507)
    val dfAgrupacionNoTipada = dfLimpio.groupBy("MERCADO", "CNAE").count().orderBy(desc("count"))
    dfAgrupacionNoTipada.show(5)


    // ==============================================================================
    // BLOQUE 3: DATASETS (API TIPADA)
    // ==============================================================================
    println("\n--> EJECUTANDO BLOQUE 3 (DATASETS)...")
    val dsConsumo = dfLimpio.as[ConsumoElectrico]

    val dsMercadoRegulado = dsConsumo.filter(cliente => cliente.MERCADO == "M1")
    val dsAnalisisImpacto = dsMercadoRegulado.map(cliente => {
      val diferenciaNoturna = cliente.ACTIVA_H22 - cliente.ACTIVA_H4
      (cliente.ANOMES, cliente.PRODUCTO, diferenciaNoturna)
    }).withColumnRenamed("_1", "ANOMES").withColumnRenamed("_2", "PRODUCTO").withColumnRenamed("_3", "Diferencia_Noche_Madrugada")

    val dsTotalHoraPuntaPorProducto = dsConsumo
      .groupByKey(cliente => cliente.PRODUCTO)
      .mapValues(cliente => cliente.ACTIVA_H14) 
      .reduceGroups((consumo_a, consumo_b) => consumo_a + consumo_b)
      .withColumnRenamed("value", "Suma_Total_H14")
    
    dsTotalHoraPuntaPorProducto.show(5)


    // ==============================================================================
    // BLOQUE 4: UDF Y SQL
    // ==============================================================================
    println("\n--> EJECUTANDO BLOQUE 4 (UDF Y SQL)...")
    
    val clasificarPerfil = udf((h19: Double, h20: Double, h21: Double, h22: Double, h23: Double) => {
      val pico = h19 + h20 + h21 + h22 + h23
      if (pico > 2500) "Consumidor EXTREMO"
      else if (pico > 1000) "Consumidor ALTO"
      else "Consumidor NORMAL"
    })

    val formatearFecha = udf((anomes: Int) => {
      val texto = anomes.toString
      texto.substring(0, 4) + "-" + texto.substring(4, 6)
    })

    val dfPerfilado = dfLimpio
      .withColumn(
        "Perfil_Intensidad_Tarde", 
        clasificarPerfil(col("ACTIVA_H19"), col("ACTIVA_H20"), col("ACTIVA_H21"), col("ACTIVA_H22"), col("ACTIVA_H23"))
      )
      .withColumn("Mes_Formateado", formatearFecha(col("ANOMES")))

    dfPerfilado.createOrReplaceTempView("tabla_consumos_electricos")

    val dfResumenSQL_A = spark.sql("""
      SELECT Mes_Formateado, Perfil_Intensidad_Tarde, COUNT(IDENTIFICADOR) as Total_Clientes, ROUND(AVG(ACTIVA_H20), 2) as Media_Consumo_H20
      FROM tabla_consumos_electricos
      GROUP BY Mes_Formateado, Perfil_Intensidad_Tarde
      ORDER BY Mes_Formateado ASC, Total_Clientes DESC
    """)

    val dfResumenSQL_B = spark.sql("""
      SELECT MERCADO, Perfil_Intensidad_Tarde, COUNT(*) as Frecuencia
      FROM tabla_consumos_electricos
      WHERE Perfil_Intensidad_Tarde != 'Consumidor NORMAL'
      GROUP BY MERCADO, Perfil_Intensidad_Tarde
      ORDER BY MERCADO ASC, Frecuencia DESC
    """)

    println("--> RESULTADO FINAL SQL A:")
    dfResumenSQL_A.show(15)

    println("--> RESULTADO FINAL SQL B:")
    dfResumenSQL_B.show(10)

    println("=====================================================")
    println(" EJECUCIÓN FINALIZADA CON ÉXITO")
    println("=====================================================")
    
    spark.stop()
  }
}
