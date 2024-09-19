#!/usr/bin/env python
# coding: utf-8

import os
os.environ['SPARK_HOME'] = 'C:\\spark'  # Replace with your Spark installation path
os.environ['JAVA_HOME'] = 'C:\\Program Files\\Java\\jdk-11'  # Replace with your Java path


from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("MyApp") \
    .config("spark.ui.showConsoleProgress", "true") \
    .getOrCreate()

# Sample PySpark DataFrame code
print(spark.range(1000).count())  # Simple count example
spark.stop()