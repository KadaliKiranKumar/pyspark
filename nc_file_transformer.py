import boto3
import botocore

from sparkxarray.reader import ncread
from pyspark.sql import SparkSession
import pyspark as spark

s3_client = boto3.client('s3')
BUCKET = 'replicated-bucket-source'
PREFIX = '20140717'

try:
    response = s3_client.list_objects(
        Bucket = BUCKET,
        Prefix = PREFIX
    )
    for file in response['Contents']:
        name = file['Key'].rsplit('/', 1)
        if name[1]:
            print(f"Downloading file {name[1]}")
            s3_client.download_file(BUCKET, file['Key'], PREFIX + '/' + name[1])
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise

sc = spark.sparkContext
paths = PREFIX + '/*.nc'
multi_rdd = ncread(sc, paths, mode='multi', partition_on=['time'], partitions=100)
multi_rdd.count()


import boto3
import socket
import random

num_executors = int(sc._conf.get('spark.executor.instances'))

def func(iterator):
    BUCKET = 'replicated-bucket-source'
    KEY = '/home/hadoop/.bashrc'
    OBJECT = socket.gethostname() + '-' + str(random.randint(1000, 10000)) + '.txt'
    print('HELO' + OBJECT)
    s3 = boto3.client('s3')
    s3.upload_file(KEY, BUCKET, OBJECT)
    return [x for x in iterator]

rdd = sc.parallelize([1, 2, 3, 4], num_executors)
rdd.mapPartitions(func).collect()

#-----------------------

BUCKET = 'replicated-bucket-source'
KEY = '/home/hadoop/.bashrc'
OBJECT = socket.gethostname() + random.randint(1, 100) + '.txt'
s3 = boto3.client('s3')
s3.upload_file(KEY, BUCKET, OBJECT)


import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import os

for filename in os.listdir('20140717'):
    if filename.endswith('.nc'):
        print(filename)
        xds = xr.open_dataset('20140717/' + filename)
        pres = xds.data_vars['air_pressure_at_sea_level'].values[:, 0, 0]
        temp = xds.data_vars['air_temperature'].values[:, 0, 0]
        times = xds.coords['time_0'].values
        pdf = pd.DataFrame({'time': times, 'pres': pres, 'temp': temp})
        pdf

sc._conf.get('spark.executor.instances')

#-----------------------
import os
from io import StringIO
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import substring
from pyspark.sql.types import IntegerType, FloatType
import time
import socket
import boto3
import botocore
import xarray as xr
import pandas as pd

BUCKET = 'replicated-bucket-source'
DATASET_NAME = 'weather'

sc = pyspark.SparkContext()
spark = SparkSession.builder.appName('netCDF').getOrCreate()

num_executors = int(sc._conf.get('spark.executor.instances'))
s3_files = []

s3 = boto3.client('s3')
# Go thru every object in the bucket
for key in s3.list_objects(Bucket=BUCKET)['Contents']:
    try:
        # Add all netCDF files to a list
        if (key['Key'].startswith('netcdf') and key['Key'].endswith('.nc')):
            s3_files.append(key['Key'])

        # Delete all CSV files from previous run
        if (key['Key'].startswith('csv') and key['Key'].endswith('.csv')):
            s3.delete_object(Bucket=BUCKET, Key=key['Key'])

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print('The object does not exist.')
        else:
            raise

# RDD of all S3 file locations that must be processed, spread into as many partitions as configured executors
rdd_input_files = sc.parallelize(s3_files, num_executors)

def process_netCDF_files(iterator):
    """Receives a list of netCDF files in S3, downloads them, extracts a subset of attributes and saves them back to S3 as CSV"""
    BUCKET = 'replicated-bucket-source'
    s3 = boto3.client('s3')
    pd_csv = pd.DataFrame()
    for s3_file_name in iterator:
        print(f'Downloading file {s3_file_name}')
        try:
            s3_dir, s3_tsdir, s3_file = s3_file_name.split('/')
            s3.download_file(BUCKET, s3_file_name, s3_file)
            xds = xr.open_dataset(s3_file)
            times = xds.coords['time_0'].values
            air_pres = (xds.data_vars['surface_air_pressure'].values[:, 0, 0] / 33.86 / 100).round(decimals=2)
            air_temp = (xds.data_vars['air_temperature'].values[:, 0, 0] - 273.15).round()
            dewpoint = (xds.data_vars['dew_point_temperature'].values[:, 0, 0] - 273.15).round()
            humidity = (xds.data_vars['relative_humidity'].values[:, 0, 0]).round()
            visibilt = (xds.data_vars['visibility_in_air'].values[:, 0, 0] * 0.00062137).round()

            # Check that feature arrays are 1-dimensional, else something is wrong with this file: discard
            if (air_pres.ndim == 1 and air_temp.ndim == 1 and dewpoint.ndim == 1 and humidity.ndim == 1 and visibilt.ndim == 1):
                pdf = pd.DataFrame({'time': times,
                                    'pres_inhg': air_pres,
                                    'temp_c': air_temp,
                                    'dewpoint_c': dewpoint,
                                    'humidity_pct': humidity,
                                    'visibility_mi': visibilt})
                pd_csv = pd_csv.append(pdf)
            else:
                print(f'File {s3_file_name} has wrong structure')

        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print('The object does not exist.')
            else:
                raise

    csv_buffer = StringIO()
    pd_csv.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_csv_file = 'csv/' + socket.gethostname() + '-' + str(int(time.time())) + '-' + str(os.getpid()) + '.csv'
    s3_resource.Object(BUCKET, s3_csv_file).put(Body=csv_buffer.getvalue())
    return [x for x in iterator]

# Process all netCDF files in parallel by sending the transformation function to all executors, along with a subset of file paths (RDD)
rdd_input_files.mapPartitions(process_netCDF_files).collect()

# By the time control arrives here, we're ready to convert generated CSV files to Parquet
ds_csv = spark.read.option('header', 'true').csv('s3://' + BUCKET + '/csv/*.csv')
ds_csv.cache() # Result is sufficiently small to fit in memory, reuse as basis for subsequent transformations

def transform_dataset(input_ds):
    part_df = input_ds.withColumn('year', substring('time', 1, 4))\
                      .withColumn('month', substring('time', 6, 2))\
                      .withColumn('day', substring('time', 9, 2))

    return part_df.withColumn('pres_inhg', part_df['pres_inhg'].cast(FloatType()))\
                      .withColumn('temp_c', part_df['temp_c'].cast(IntegerType()))\
                      .withColumn('dewpoint_c', part_df['dewpoint_c'].cast(IntegerType()))\
                      .withColumn('humidity_pct', part_df['humidity_pct'].cast(IntegerType()))\
                      .withColumn('visibility_mi', part_df['visibility_mi'].cast(IntegerType()))

# Generate
final_ds = transform_dataset(ds_csv)
final_ds.write.mode('overwrite').partitionBy('year', 'month', 'day').parquet('s3://' + BUCKET + '/parquet/' + DATASET_NAME)

sc.stop()

# --------

pdf = pd.DataFrame({'time': times, 'pres': pres, 'temp': temp})
pdf.to_csv(index=False)

DOWNLOAD_DIR = '/home/hadoop'

local_dir = DOWNLOAD_DIR + '/' + s3_dir
if not os.path.exists(local_dir):
    os.makedirs(local_dir)


s3 = boto3.client('s3')
for key in s3.list_objects(Bucket=BUCKET)['Contents']:
    if (key['Key'].startswith('csv') and key['Key'].endswith('.csv')):
        print(key['Key'])
        s3.delete_object(Bucket=BUCKET, Key=key['Key'])

times = xds.coords['time_0'].values
wind_spd = (xds.data_vars['x_wind'].values[:, 0, 0] * 1.943844).round()
wind_gst = (xds.data_vars['wind_speed_of_gust'].values[:, 0, 0] * 1.943844).round()
air_pres = (xds.data_vars['surface_air_pressure'].values[:, 0, 0] / 33.86 / 100).round(decimals=2)
air_temp = (xds.data_vars['air_temperature'].values[:, 0, 0] - 273.15).round()
dewpoint = (xds.data_vars['dew_point_temperature'].values[:, 0, 0] - 273.15).round()
humidity = (xds.data_vars['relative_humidity'].values[:, 0, 0].round()
visibilt = (xds.data_vars['visibility_in_air'].values[:, 0, 0] * 0.00062137).round()

pdf = pd.DataFrame({'time': times, 'wind_kts': wind_spd, 'gust_kts': wind_gst, 'pres_inhg': air_pres, 'temp_c': air_temp, 'dewpoint_c': dewpoint, 'humidity_pct': humidity, 'visibility_mi': visibilt})




pd.DataFrame([['a','1'],['b','2']],
                   dtype={'x':'object','y':'int'},
                   columns=['x','y'])