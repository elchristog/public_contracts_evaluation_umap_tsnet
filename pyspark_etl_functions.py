import os
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (col, count, when, isnan, isnull, min, max, mean, stddev, sum, avg, 
                                   month, year, months_between, current_date, concat, lit)

import pyspark.sql.functions as F




def read_csv_with_pyspark(filename, folder='data', separator='|'):
    spark = SparkSession.builder \
        .appName("Read CSV with PySpark") \
        .getOrCreate()

    file_path = os.path.join(folder, filename)
    df = spark.read.csv(file_path, header=True, inferSchema=True, sep=separator)
    
    return df


def analyze_data_quality(df):
    spark = SparkSession.builder \
        .appName("Analyze Data Quality with PySpark") \
        .getOrCreate()

    # Cuenta de registros
    record_count = df.count()
    print(f"Total de registros: {record_count}")

    # Cuenta de registros nulos y duplicados
    null_count = df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).collect()
    duplicate_count = df.count() - df.dropDuplicates().count()

    print("\nConteo de registros nulos por columna:")
    for col_name, nulls in zip(df.columns, null_count[0]):
        print(f"{col_name}: {nulls}")

    print(f"\nConteo de registros duplicados: {duplicate_count}")

    # Estadísticas descriptivas (mínimo, máximo, promedio, desviación estándar) para columnas numéricas
    numeric_columns = [col_name for col_name, dtype in df.dtypes if dtype in ("double", "float", "int", "bigint", "smallint")]
    summary_stats = df.select(numeric_columns).summary("min", "max", "mean", "stddev").collect()

    print("\nEstadísticas descriptivas para columnas numéricas:")
    for stat in summary_stats:
        print(f"{stat['summary']}:")

        for col_name in numeric_columns:
            print(f"  {col_name}: {stat[col_name]}")

    # Identificación de valores atípicos (basado en el rango intercuartil)
    for col_name in numeric_columns:
        q1, q3 = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df.filter((col(col_name) < lower_bound) | (col(col_name) > upper_bound)).count()

        print(f"\nValores atípicos en la columna {col_name}: {outliers}")



def limpiar_nulos_y_duplicados(df: DataFrame, columnas: list) -> DataFrame:
    df_limpio = df.dropna(subset=columnas)

    df_limpio = df_limpio.dropDuplicates()

    return df_limpio


def filtrar_ultimos_anos(df, years=2):
    today_year = datetime.today().year
    df_filtrado = df.filter((df.year >= today_year - years) & (df.year <= today_year))
    return df_filtrado

def seleccionar_columnas(df, columnas):
    return df.select(columnas)


def agregar_por_id_contratista(df):
    agg_exprs = [
        F.count("*").alias("num_contratos"),
        F.sum("VALOR_TOTAL_CONTRATO").alias("suma_valor_total_contrato"),
        F.avg("VALOR_TOTAL_CONTRATO").alias("promedio_valor_total_contrato"),
        F.max(F.struct("year", "month")).alias("ultimo_contrato"),
        F.countDistinct("DEPARTAMENTO").alias("num_departamentos"),
        F.countDistinct("ESTADO_DEL_PROCESO").alias("num_estados_proceso"),
        F.countDistinct("CLASE_PROCESO").alias("num_clases_proceso"),
        F.countDistinct("TIPO_PROCESO").alias("num_tipos_proceso"),
        F.countDistinct("NOMBRE_FAMILIA").alias("num_familias"),
        F.countDistinct("NOMBRE_CLASE").alias("num_clases")
    ]
    
    df_agregado = (
        df.groupBy("ID_CONTRATISTA", "RAZON_SOCIAL_CONTRATISTA")
        .agg(*agg_exprs)
        .withColumn(
            "meses_desde_ultimo_contrato",
            (F.year(F.current_date()) - F.col("ultimo_contrato.year")) * 12
            + (F.month(F.current_date()) - F.col("ultimo_contrato.month"))
        )
        .drop("ultimo_contrato")
        .orderBy(F.desc("num_contratos")) 
    )
    
    return df_agregado


def agregar_por_nit_entidad(df):
    agg_exprs = [
        F.count("*").alias("num_contratos"),
        F.sum("VALOR_TOTAL_CONTRATO").alias("suma_valor_total_contrato"),
        F.avg("VALOR_TOTAL_CONTRATO").alias("promedio_valor_total_contrato"),
        F.max(F.struct("year", "month")).alias("ultimo_contrato"),
        F.countDistinct("DEPARTAMENTO").alias("num_departamentos"),
        F.countDistinct("ESTADO_DEL_PROCESO").alias("num_estados_proceso"),
        F.countDistinct("CLASE_PROCESO").alias("num_clases_proceso"),
        F.countDistinct("TIPO_PROCESO").alias("num_tipos_proceso"),
        F.countDistinct("NOMBRE_FAMILIA").alias("num_familias"),
        F.countDistinct("NOMBRE_CLASE").alias("num_clases")
    ]
    
    df_agregado = (
        df.groupBy("NIT_ENTIDAD", "NOMBRE_ENTIDAD")
        .agg(*agg_exprs)
        .withColumn(
            "meses_desde_ultimo_contrato",
            (F.year(F.current_date()) - F.col("ultimo_contrato.year")) * 12
            + (F.month(F.current_date()) - F.col("ultimo_contrato.month"))
        )
        .drop("ultimo_contrato")
        .orderBy(F.desc("num_contratos")) 
    )
    
    return df_agregado


def pivotar_por_columna(df, columna):
    valores = df.select(columna).distinct().rdd.flatMap(lambda x: x).collect()

    df_pivote = (
        df.groupBy("NIT_ENTIDAD", "NOMBRE_ENTIDAD")
        .pivot(columna, valores)
        .count()
        .fillna(0)
    )
    
    return df_pivote

def unir_dataframes(df1, df2):
    df_unido = df1.join(df2, on=["NIT_ENTIDAD", "NOMBRE_ENTIDAD"], how="inner")
    return df_unido

def aggregate_multas_data(multas_df):
    result = multas_df.groupBy("nit_entidad").agg(
        count("*").alias("numero_de_multas"),
        sum("valor_sancion").alias("suma_valor_sancion"),
        avg("valor_sancion").alias("promedio_valor_sancion"),
        months_between(current_date(), max("fecha_de_publicacion")).alias("meses_desde_ultima_multa")
    )
    # result = result.withColumn("nit_entidad", concat(result["nit_entidad"], lit("0")))
    return result

def left_join_dataframes(df1, df2, df1_key, df2_key):
    joined_df = df1.join(df2, df1[df1_key] == df2[df2_key], how="left")
    return joined_df