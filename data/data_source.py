from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType

import pandas as pd
import numpy as np
import configparser


class DataSource:
    def __init__(self):
        """Create a spark context and session"""

        config = configparser.ConfigParser()
        config.read('./data/data_source.ini')
        master_url = config['spark']['master_url']
        app_name = config['spark']['app_name']
        print(f"Creating Spark context to {master_url}...")

        conf = SparkConf().setAppName(app_name) \
                          .setMaster(master_url) \
                          .set(key="spark.sql.execution.arrow.enabled", value="true")

        self.sc = SparkContext(conf=conf)
        self.spark = SparkSession.builder \
                                 .config(conf=conf) \
                                 .getOrCreate()

        print("Created Spark context...")

    def get_data(self, num_elements=1000) -> pd.DataFrame:
        """Create sample data, apply Python logic, return Pandas dataframe"""

        # Create RDD from a vector containing a normal distribution
        mu, sigma = 2, 0.5
        v = np.random.normal(mu, sigma, num_elements)
        rdd = self.sc.parallelize(v)
        print(f"RDD containing {rdd.count()} elements and {rdd.getNumPartitions()} partitions")

        # Apply a transformation that multiplies the RDD contents by 2, also convert numpy.float64 to Python float
        def mult(x): return x * np.array([2])
        rdd2 = rdd.map(mult).map(lambda x: (float(x),))

        # Convert RDD into PySpark dataframe, and define a view on it
        schema = StructType([StructField("value", FloatType(), True)])
        df1 = self.spark.createDataFrame(rdd2, schema)
        df1.registerTempTable("test")

        # Define a Python UDF
        def square(x): return x ** 2
        self.spark.udf.register("squaredWithPython", square)

        # Query the view using the UDF
        df2 = self.spark.sql("SELECT squaredWithPython(value) AS value_squared FROM test")
        print("Spark dataset top 10 rows:")
        df2.show(10)

        # Return the result as Pandas dataframe
        return df2.toPandas()
