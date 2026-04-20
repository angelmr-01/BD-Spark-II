# Entrega 3: Procesamiento Big Data con Apache Spark

Este repositorio contiene la implementación de un pipeline ETL completo para el análisis de consumos eléctricos utilizando Apache Spark y Scala. El proyecto aborda la limpieza, transformación y análisis algorítmico de grandes volúmenes de datos mediante diferentes APIs de Spark.

## Objetivos del Proyecto
- Procesamiento paralelo a bajo nivel utilizando RDD.
- Manipulación tabular y operaciones masivas mediante DataFrames (API No Tipada).
- Validación estricta de tipos de datos utilizando Datasets (API Tipada).
- Generación de reportes analíticos utilizando Spark SQL y funciones definidas por el usuario (UDFs).

## Estructura del Repositorio
- `notebooks/`: Contiene el código interactivo (`codigo_zeppelin.scala`) diseñado para exploración e iteración visual en un entorno Apache Zeppelin.
- `trabajo/`: Contiene el código fuente productivo (`PracticaSpark.scala`) y la configuración `build.sbt` para la generación del ejecutable.
- `docker-compose.yml`: Archivo de infraestructura para desplegar un entorno completo de clúster Spark (Master, Workers, Client) y Zeppelin.

## Tecnologías Utilizadas
- Lenguaje: Scala 2.12
- Framework: Apache Spark 3.5.5 (Core & SQL)
- Gestor de Dependencias: SBT
- Despliegue: Docker & Docker Compose

## Ejecución del Proyecto

1. Compilación del código fuente:
Para generar el artefacto `.jar` necesario para el entorno de producción, acceda a la carpeta `trabajo/` y ejecute el paquete SBT (puede realizarse mediante entorno IDE o consola).

2. Despliegue en Clúster:
Se debe iniciar el entorno virtual y enviar el trabajo al nodo Master mediante el contenedor cliente:
```bash
docker-compose up -d
docker exec -it spark-client bash
/opt/spark/bin/spark-submit --master spark://spark-master:7077 --class es.upm.bd.PracticaSpark /home/trabajo/target/scala-2.12/entrega3-bigdata_2.12-1.0.jar
```
