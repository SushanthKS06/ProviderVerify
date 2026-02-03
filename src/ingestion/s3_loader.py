"""
S3 data loader for ProviderVerify.

Loads provider data from S3-compatible storage with support for CSV, Parquet,
and JSON formats. Includes error handling and logging for compliance.
"""

import logging
import os
from typing import Dict, List, Optional, Union
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError, NoCredentialsError
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.types import StructType, StructField, StringType

logger = logging.getLogger(__name__)


class S3DataLoader:
    """
    Handles loading provider data from S3-compatible storage.
    
    Security note: Ensure proper IAM roles and encryption are configured
    for S3 buckets containing PHI data.
    """
    
    def __init__(self, bucket_name: str, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = "us-east-1"):
        """
        Initialize S3 loader with credentials.
        
        Args:
            bucket_name: S3 bucket name
            aws_access_key_id: AWS access key (optional if using IAM roles)
            aws_secret_access_key: AWS secret key (optional if using IAM roles)
            region_name: AWS region
        """
        self.bucket_name = bucket_name
        self.region_name = region_name
        
        # Initialize S3 client with credentials or IAM roles
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
            else:
                # Use IAM roles or environment variables
                self.s3_client = boto3.client('s3', region_name=region_name)
                
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure credentials.")
            raise
        except ClientError as e:
            logger.error(f"Failed to connect to S3 bucket {bucket_name}: {e}")
            raise
    
    def list_files(self, prefix: str = "", file_extension: Optional[str] = None) -> List[str]:
        """
        List files in S3 bucket with optional prefix and extension filtering.
        
        Args:
            prefix: S3 prefix to filter files
            file_extension: File extension to filter (e.g., 'csv', 'parquet')
            
        Returns:
            List of S3 object keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    if file_extension:
                        if key.lower().endswith(f'.{file_extension.lower()}'):
                            files.append(key)
                    else:
                        files.append(key)
            
            logger.info(f"Found {len(files)} files with prefix '{prefix}'")
            return files
            
        except ClientError as e:
            logger.error(f"Failed to list files in bucket {self.bucket_name}: {e}")
            raise
    
    def load_file(self, s3_key: str, file_format: str = "csv",
                  spark_session: Optional[SparkSession] = None) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Load a single file from S3.
        
        Args:
            s3_key: S3 object key
            file_format: File format ('csv', 'parquet', 'json')
            spark_session: Spark session for Spark DataFrame (optional)
            
        Returns:
            Pandas DataFrame or Spark DataFrame
        """
        try:
            if file_format.lower() == "csv":
                return self._load_csv(s3_key, spark_session)
            elif file_format.lower() == "parquet":
                return self._load_parquet(s3_key, spark_session)
            elif file_format.lower() == "json":
                return self._load_json(s3_key, spark_session)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
        except Exception as e:
            logger.error(f"Failed to load file {s3_key}: {e}")
            raise
    
    def _load_csv(self, s3_key: str, spark_session: Optional[SparkSession] = None) -> Union[pd.DataFrame, SparkDataFrame]:
        """Load CSV file from S3."""
        if spark_session:
            # Load with Spark
            df = spark_session.read.csv(
                f"s3a://{self.bucket_name}/{s3_key}",
                header=True,
                inferSchema=True,
                nullValue="",
                emptyValue=""
            )
            logger.info(f"Loaded CSV {s3_key} as Spark DataFrame with {df.count()} rows")
            return df
        else:
            # Load with Pandas
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            df = pd.read_csv(obj['Body'])
            logger.info(f"Loaded CSV {s3_key} as Pandas DataFrame with {len(df)} rows")
            return df
    
    def _load_parquet(self, s3_key: str, spark_session: Optional[SparkSession] = None) -> Union[pd.DataFrame, SparkDataFrame]:
        """Load Parquet file from S3."""
        if spark_session:
            # Load with Spark
            df = spark_session.read.parquet(f"s3a://{self.bucket_name}/{s3_key}")
            logger.info(f"Loaded Parquet {s3_key} as Spark DataFrame with {df.count()} rows")
            return df
        else:
            # Load with Pandas/PyArrow
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            table = pq.read_table(obj['Body'])
            df = table.to_pandas()
            logger.info(f"Loaded Parquet {s3_key} as Pandas DataFrame with {len(df)} rows")
            return df
    
    def _load_json(self, s3_key: str, spark_session: Optional[SparkSession] = None) -> Union[pd.DataFrame, SparkDataFrame]:
        """Load JSON file from S3."""
        if spark_session:
            # Load with Spark
            df = spark_session.read.json(f"s3a://{self.bucket_name}/{s3_key}")
            logger.info(f"Loaded JSON {s3_key} as Spark DataFrame with {df.count()} rows")
            return df
        else:
            # Load with Pandas
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            df = pd.read_json(obj['Body'], lines=True)
            logger.info(f"Loaded JSON {s3_key} as Pandas DataFrame with {len(df)} rows")
            return df
    
    def load_multiple_files(self, s3_keys: List[str], file_format: str = "csv",
                           spark_session: Optional[SparkSession] = None) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Load and concatenate multiple files from S3.
        
        Args:
            s3_keys: List of S3 object keys
            file_format: File format for all files
            spark_session: Spark session (optional)
            
        Returns:
            Concatenated DataFrame
        """
        if not s3_keys:
            logger.warning("No files provided to load")
            return pd.DataFrame() if not spark_session else spark_session.createDataFrame([], StructType([]))
        
        dataframes = []
        for s3_key in s3_keys:
            try:
                df = self.load_file(s3_key, file_format, spark_session)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Failed to load {s3_key}: {e}")
                continue
        
        if not dataframes:
            logger.error("No files were successfully loaded")
            return pd.DataFrame() if not spark_session else spark_session.createDataFrame([], StructType([]))
        
        if spark_session:
            # Concatenate Spark DataFrames
            result_df = dataframes[0]
            for df in dataframes[1:]:
                result_df = result_df.union(df)
            logger.info(f"Concatenated {len(dataframes)} Spark DataFrames with {result_df.count()} total rows")
            return result_df
        else:
            # Concatenate Pandas DataFrames
            result_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Concatenated {len(dataframes)} Pandas DataFrames with {len(result_df)} total rows")
            return df
    
    def save_to_s3(self, df: Union[pd.DataFrame, SparkDataFrame], s3_key: str,
                   file_format: str = "parquet", compression: str = "snappy") -> bool:
        """
        Save DataFrame to S3.
        
        Args:
            df: DataFrame to save
            s3_key: Destination S3 key
            file_format: Output format ('csv', 'parquet')
            compression: Compression codec
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if isinstance(df, SparkDataFrame):
                # Save Spark DataFrame
                if file_format.lower() == "parquet":
                    df.write.mode("overwrite").parquet(
                        f"s3a://{self.bucket_name}/{s3_key}",
                        compression=compression
                    )
                elif file_format.lower() == "csv":
                    df.write.mode("overwrite").csv(
                        f"s3a://{self.bucket_name}/{s3_key}",
                        header=True,
                        compression=compression
                    )
            else:
                # Save Pandas DataFrame
                if file_format.lower() == "parquet":
                    buffer = pa.compress_dataframe(df, compression=compression)
                    self.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        Body=buffer.to_pybytes()
                    )
                elif file_format.lower() == "csv":
                    csv_buffer = df.to_csv(index=False)
                    self.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        Body=csv_buffer.encode('utf-8')
                    )
            
            logger.info(f"Successfully saved DataFrame to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save DataFrame to S3: {e}")
            return False


def create_spark_session(app_name: str = "ProviderVerify", master: str = "local[*]") -> SparkSession:
    """
    Create Spark session with S3 configuration.
    
    Args:
        app_name: Spark application name
        master: Spark master URL
        
    Returns:
        Configured Spark session
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    logger.info("Created Spark session with S3 configuration")
    return spark
