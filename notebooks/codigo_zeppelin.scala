// ==============================================================================
// BLOQUE 1: LECTURA Y LIMPIEZA CON SPARK CORE (API RDD)
// ==============================================================================
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

// 1. LECTURA DEL FICHERO USANDO RDD
// Como el dataset no tiene cabecera, leemos directamente el texto
val rddCrudo = spark.sparkContext.textFile("file:///opt/spark/data/endesaAgregada")

// 2. SELECCIÓN DE COLUMNAS (Ignorar reactiva y H25)
// separamos por comas y extraemos sólo la información base y las horas del 1 al 24.
val rddSeleccionado = rddCrudo.map(linea => {
  val cols = linea.split(",")
  val identificador = cols(0).trim
  val anomes = cols(1).trim
  val cnae = cols(2).trim
  val producto = cols(3).trim
  val mercado = cols(4).trim
  
  // Extraemos H1 a H24 (posiciones 5 a 28), ignorando H25 (29) y Reactivas (30+)
  val consumosArray = cols.slice(5, 29).map(_.trim.toDouble)
  
  (identificador, anomes, cnae, producto, mercado, consumosArray)
})

// 3. PREPROCESAMIENTO Y LIMPIEZA DE DATOS
// Tras seleccionar columnas, aplicamos limpieza eliminando cualquier fila
// donde el CNAE o PRODUCTO vengan vacíos, o si existen mediciones negativas erróneas.
val rddLimpio = rddSeleccionado.filter { caso => 
  caso._3.nonEmpty && caso._4.nonEmpty && caso._6.forall(_ >= 0)
}

// 4. APLICACIÓN DE MÉTODOS SOBRE RDDs (PARALELOS)
// Transformación Paralela 1 (Map): Calculamos de nuevo la estructura y la suma total de
// consumos.
val rddProcesado = rddLimpio.map(caso => {
  val (identificador, anomes, cnae, producto, mercado, consumosArray) = caso
  val totalActiva = consumosArray.sum
  (identificador, anomes, cnae, producto, mercado, totalActiva)
})

// Transformación Paralela 2 (Shuffling / Agrupación): Calculamos el consumo total por tipo de mercado
// extraemos (Mercado -> Valor), y sumamos todo usando reduceByKey en paralelo en todo el cluster.
val rddTotalPorMercado = rddProcesado
  .map(x => (x._5, x._6)) // x._5 es el mercado, x._6 es el totalActiva
  .reduceByKey(_ + _)

// Disparamos la ejecución
println("Consumo global por mercado:")
rddTotalPorMercado.collect().foreach(println)


// ==============================================================================
// BLOQUE 2: LECTURA, SELECCIÓN, LIMPIEZA Y CONSULTAS (API NO TIPADA - DataFrames)
// ==============================================================================

// 1. LECTURA
// Definimos el mismo esquema (5 base + 25 activa + 25 reactiva)
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

// 2. SELECCIÓN DE COLUMNAS
// Eliminamos explícitamente H25 y Reactivas mediante la API de Dataframe
val columnasABorrar = Seq("ACTIVA_H25") ++ (1 to 25).map(i => s"REACTIVA_H$i")
val dfSeleccionado = dfBruto.drop(columnasABorrar: _*)

// 3. PREPROCESAMIENTO: LIMPIEZA DE DATOS
// Eliminamos campos clave nulos y aplicamos la misma lógica que en el RDD:
// Filtramos para asegurar que ninguna de las 24 horas contenga consumos negativos.
val condicionPositivos = (1 to 24).map(i => col(s"ACTIVA_H$i") >= 0).reduce(_ && _)
val dfLimpio = dfSeleccionado.na.drop(Seq("IDENTIFICADOR", "CNAE", "MERCADO")).filter(condicionPositivos)

// 4. VARIAS CONSULTAS CON API NO TIPADA
// Consulta A: Mostrar los datos base de usuarios concretos correspondientes al mes de Julio de 2015 (201507)
val dfFiltroAnomes = dfLimpio.filter(col("ANOMES") === 201507)
println("Consulta A (No Tipada) - Filtro Clientes de Julio 2015:")
dfFiltroAnomes.select("IDENTIFICADOR", "MERCADO", "PRODUCTO", "ACTIVA_H1", "ACTIVA_H20").show(5)

// Consulta B: Contar volumen total de clientes repartidos en su respectivo mercado y perfil de usuario (CNAE)
val dfAgrupacionNoTipada = dfLimpio.groupBy("MERCADO", "CNAE").count().orderBy(desc("count"))
println("Consulta B (No Tipada) - Agrupación y Conteo por Mercado y Perfil:")
dfAgrupacionNoTipada.show()



// ==============================================================================
// BLOQUE 3: CONSULTAS CON API TIPADA (Datasets)
// ==============================================================================


// BLOQUE 3.1: DEFINICIÓN DE CLASES
// En Zeppelin, las Case Classes deben definirse en una celda separada antes de usarlas,
// de lo contrario el compilador lanza un "NullPointerException OuterScope". Hay que 
// copiarlo solo en una nueva celda, ejecutarlo, y luego pasar a la siguiente parte.

case class ConsumoElectrico(
  IDENTIFICADOR: String, ANOMES: Int, CNAE: String, PRODUCTO: String, MERCADO: String,
  ACTIVA_H1: Double, ACTIVA_H2: Double, ACTIVA_H3: Double, ACTIVA_H4: Double, ACTIVA_H5: Double,
  ACTIVA_H6: Double, ACTIVA_H7: Double, ACTIVA_H8: Double, ACTIVA_H9: Double, ACTIVA_H10: Double,
  ACTIVA_H11: Double, ACTIVA_H12: Double, ACTIVA_H13: Double, ACTIVA_H14: Double, ACTIVA_H15: Double,
  ACTIVA_H16: Double, ACTIVA_H17: Double, ACTIVA_H18: Double, ACTIVA_H19: Double, ACTIVA_H20: Double,
  ACTIVA_H21: Double, ACTIVA_H22: Double, ACTIVA_H23: Double, ACTIVA_H24: Double
)


import spark.implicits._

// Transformación a Dataset (El compilador valida los tipos automáticamente desde el DataFrame limpio)
val dsConsumo = dfLimpio.as[ConsumoElectrico]

// 2. VARIAS CONSULTAS CON API TIPADA
// Consulta A: Diferencia de impacto entre horario nocturno (22h) y madrugada (4h) en Mercado M1
val dsMercadoRegulado = dsConsumo.filter(cliente => cliente.MERCADO == "M1")

val dsAnalisisImpacto = dsMercadoRegulado.map(cliente => {
  val diferenciaNoturna = cliente.ACTIVA_H22 - cliente.ACTIVA_H4
  // Devolvemos tuplas con la forma (AñoMes, Producto, Diferencia de Consumo)
  (cliente.ANOMES, cliente.PRODUCTO, diferenciaNoturna)
}).withColumnRenamed("_1", "ANOMES").withColumnRenamed("_2", "PRODUCTO").withColumnRenamed("_3", "Diferencia_Noche_Madrugada")

println("Consulta A (Tipada) - Consumo Nocturno frente a Madrugada en Mercado M1:")
dsAnalisisImpacto.orderBy(desc("Diferencia_Noche_Madrugada")).show(5)

// Consulta B: Agrupación estrictamente Tipada (groupByKey). 
// Sumario de la Hora de comer (ACTIVA_H14) categorizado por producto.
val dsTotalHoraPuntaPorProducto = dsConsumo
  .groupByKey(cliente => cliente.PRODUCTO)
  .mapValues(cliente => cliente.ACTIVA_H14) 
  .reduceGroups((consumo_a, consumo_b) => consumo_a + consumo_b)
  .withColumnRenamed("value", "Suma_Total_H14")

println("Consulta B (Tipada) - Suma total de consumo a la Hora de Comer (H14) por Producto:")
dsTotalHoraPuntaPorProducto.show()


// ==============================================================================
// BLOQUE 4: UDFs, REGISTRO DE TABLAS Y API SQL
// ==============================================================================

// 1. REALIZACIÓN DE UDF ADECUADAS
// UDF A (Segmentación): Analiza las horas de la tarde-noche (Prime Time: 19h a 23h)
// y determina el perfil de intensidad de consumo del cliente en esa franja horaria.
val clasificarPerfil = udf((h19: Double, h20: Double, h21: Double, h22: Double, h23: Double) => {
  val pico = h19 + h20 + h21 + h22 + h23
  if (pico > 2500) "Consumidor EXTREMO"
  else if (pico > 1000) "Consumidor ALTO"
  else "Consumidor NORMAL"
})

// UDF B (Manipulación de Strings): Corta el número 201507 para separarlo visualmente en "2015-07"
val formatearFecha = udf((anomes: Int) => {
  val texto = anomes.toString
  texto.substring(0, 4) + "-" + texto.substring(4, 6)
})

// Aplicamos ambas UDFs al conjunto de datos de forma paralela en la proyección de datos
val dfPerfilado = dfLimpio
  .withColumn(
    "Perfil_Intensidad_Tarde", 
    clasificarPerfil(col("ACTIVA_H19"), col("ACTIVA_H20"), col("ACTIVA_H21"), col("ACTIVA_H22"), col("ACTIVA_H23"))
  )
  .withColumn("Mes_Formateado", formatearFecha(col("ANOMES")))

// 2. REGISTRO DE TABLAS 
// Registramos el DataFrame enriquecido con la UDF como una tabla global temporal en memoria
dfPerfilado.createOrReplaceTempView("tabla_consumos_electricos")

// 3. VARIAS CONSULTAS ADECUADAS CON API SQL
// Consulta SQL A: Ejecutamos una analítica agrupando por el mes formateado generado por UDF y perfiles
val dfResumenSQL_A = spark.sql("""
  SELECT Mes_Formateado, Perfil_Intensidad_Tarde, COUNT(IDENTIFICADOR) as Total_Clientes, ROUND(AVG(ACTIVA_H20), 2) as Media_Consumo_H20
  FROM tabla_consumos_electricos
  GROUP BY Mes_Formateado, Perfil_Intensidad_Tarde
  ORDER BY Mes_Formateado ASC, Total_Clientes DESC
""")

println("Resultado Análisis SQL (A) - Agrupación de Perfiles Punta:")
dfResumenSQL_A.show(15)

// Consulta SQL B: Cruzamos los perfiles identificados por nuestra UDF con los tipos de Mercado
val dfResumenSQL_B = spark.sql("""
  SELECT MERCADO, Perfil_Intensidad_Tarde, COUNT(*) as Frecuencia
  FROM tabla_consumos_electricos
  WHERE Perfil_Intensidad_Tarde != 'Consumidor NORMAL'
  GROUP BY MERCADO, Perfil_Intensidad_Tarde
  ORDER BY MERCADO ASC, Frecuencia DESC
""")

println("Resultado Análisis SQL (B) - Frecuencia de Picos Anormales en cada Mercado:")
dfResumenSQL_B.show(10)

// ==============================================================================
// BLOQUE 5: VISUALIZACIÓN DE RESULTADOS INTERACTIVOS
// ==============================================================================

// 1. Gráfica del Bloque 1 (RDDs):
// Convertimos el RDD agrupado a Dataframe al vuelo para que Zeppelin pueda graficarlo
import spark.implicits._
val dfRddVisual = rddTotalPorMercado.toDF("Tipo_Mercado", "Total_Suma_Activa")
z.show(dfRddVisual)

// 2. Gráfica del Bloque 2 (DataFrame No Tipado):
z.show(dfAgrupacionNoTipada)

// 3. Gráfica del Bloque 3 (Datasets Tipados):
z.show(dsTotalHoraPuntaPorProducto)

// 4. Gráficas del Bloque 4 (SQL y UDFs):
z.show(dfResumenSQL_A)
z.show(dfResumenSQL_B)
