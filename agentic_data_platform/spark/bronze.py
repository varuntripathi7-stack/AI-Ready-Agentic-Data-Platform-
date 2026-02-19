#!/usr/bin/env python3
"""
Bronze Layer - Raw Data Ingestion
Reads streaming data from Kafka and writes raw data to Delta Lake.
NO filtering, NO cleaning - stores everything as-it is.
"""

import os
# Ensure compatible Java version for Spark
if os.path.isdir("/usr/lib/jvm/java-17-openjdk-amd64"):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"

# # SparkSession is the entry point to Spark.
# It starts the Spark engine and lets us read, process, and write data using DataFrames.
from pyspark.sql import SparkSession  


# Used to parse JSON strings into structured columns
# col()	-> Refer to DataFrame columns safely
# from_json() ->	Convert JSON string → structured columns
from pyspark.sql.functions import col, from_json # Used to parse JSON strings into structured columns



# here We are telling Spark what columns exist and what type of data each column holds.
# all below are spark built - in classes which is used to define schema for our data
# StructType → represents the whole schema (table structure) 

# StructField → represents one column

# IntegerType → integer values (e.g. user_id)

# StringType → text values (e.g. event_type)

# DoubleType → decimal numbers (e.g. price)

from pyspark.sql.types import ( 
    StructType, StructField, IntegerType, StringType, DoubleType
)


# Kafka Configuration 
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "ecommerce_events"





# Here we are defining the paths for bronze data storage and Spark checkpointing
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get the base directory of the project
BRONZE_PATH = os.path.join(BASE_PATH, "data/bronze/ecommerce_events") # Path to store raw bronze data
CHECKPOINT_PATH = os.path.join(BASE_PATH, "data/checkpoints/bronze") # Path for Spark checkpointing


# Here We are telling Spark what fields exist in the JSON event and what type of data each field contains.
# Event Schema matching the generator 
EVENT_SCHEMA = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("event_type", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("timestamp", StringType(), True)
])






# This function starts Spark and prepares it to read streaming data from Kafka 
# and write data to Delta Lake, using the required configurations and dependencies.
def create_spark_session() -> SparkSession: 
    """
    Create and configure Spark session with Delta Lake and Kafka support.
    """
    # Building a Spark session with necessary configurations for Delta Lake and Kafka integration.
    # Note -> We are using Delta Lake to store raw streaming data from Kafka in a 
    # safe, reliable, and replayable way.

    # Configurations include:
    # - Enabling Delta Lake SQL extensions
    # - Setting the catalog to use Delta Lake
    # - Adding necessary JAR packages for Kafka and Delta Lake support      
    # - Setting the checkpoint location for streaming queries
    # Finally, we set the log level to WARN to reduce verbosity.
    # Returns the configured Spark session.

    #Note:-   
    #  .appName -> Sets the name of the Spark application.
    #  .config -> Configures various Spark settings.
    #  .getOrCreate() -> Creates the Spark session if it doesn't exist, otherwise returns the existing one.
    spark = SparkSession.builder \
        .appName("Bronze_Layer_Ingestion") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "io.delta:delta-spark_2.12:3.1.0") \
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_PATH) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN") # Set log level to WARN to reduce verbosity
    return spark 








def read_kafka_stream(spark: SparkSession): 
    """
    This function connects Spark to Kafka and continuously reads messages from a Kafka topic 
    as a streaming DataFrame.

    Read streaming data from Kafka topic.
    """
    # Print Kafka connection details
    
    print(f"Reading from Kafka topic: {KAFKA_TOPIC}")

    # Creating DataFrame representing the stream of input lines from Kafka
    # .readStream = reading streaming source 
    # .format("kafka") = specifying Kafka as the source format (Tells Spark: “Data is coming from Kafka”)
    # .option(...) = setting various options for Kafka connection
    # .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) = specifies the Kafka server address (Tells Spark: where Kafka is running and how to connect)
    # .option("subscribe", KAFKA_TOPIC) = specifies the Kafka topic to read (Tells Spark: which topic to read messages from)
    # .option("startingOffsets", "earliest") = starts reading from the earliest available message (Tells Spark: read all messages from the beginning)
    # .option("failOnDataLoss", "false") = prevents failure if some data is lost (Tells Spark: continue processing even if some messages are missing)
    # .load() = actually loading the data into a DataFrame (Tells Spark: start reading the data stream)
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()
    
    return kafka_df







def parse_events(kafka_df):
    """
    This function converts raw Kafka messages (binary JSON) into structured columns 
    while keeping the data raw and unchanged.

    Parse JSON events from Kafka messages.
    Keeps raw data - no filtering or cleaning.
    """
    # Convert binary value to string and parse JSON


    # .selectExpr() is spark method which allows us to write SQL expressions inside DataFrame code (selectExpr = SQL SELECT inside DataFrame API)
    # Here we are selecting and transforming the raw Kafka data:
    # - CAST(key AS STRING) as kafka_key: Converts the binary key to a string and aliases it as kafka_key
    # - CAST(value AS STRING) as json_value: Converts the binary value to a string and aliases it as json_value
    # - topic, partition, offset, timestamp as kafka_timestamp: Selects the topic, partition, offset, and timestamp from Kafka
    # Then we use from_json() to parse the json_value column using the defined EVENT_SCHEMA
    # Finally, we select the individual fields from the parsed JSON and keep the Kafka metadata columns 
    parsed_df = kafka_df \
        .selectExpr("CAST(key AS STRING) as kafka_key",
                    "CAST(value AS STRING) as json_value",
                    "topic",
                    "partition",
                    "offset",
                    "timestamp as kafka_timestamp") \
        .select(
            col("kafka_key"),
            from_json(col("json_value"), EVENT_SCHEMA).alias("event"),
            col("topic"),
            col("partition"),
            col("offset"),
            col("kafka_timestamp")
        ) \
        .select(
            col("kafka_key"),
            col("event.user_id").alias("user_id"),
            col("event.product_id").alias("product_id"),
            col("event.event_type").alias("event_type"),
            col("event.price").alias("price"),
            col("event.timestamp").alias("event_timestamp"),
            col("topic"),
            col("partition"),
            col("offset"),
            col("kafka_timestamp")
        ) # Here We are flattening the parsed JSON event into individual columns and keeping Kafka metadata for reliability.
    
    return parsed_df








def write_to_delta(df, output_path: str, checkpoint_path: str):
    """
    This function continuously writes the processed streaming data into Delta Lake 
    in a safe and reliable way.

    This function:
       takes a Spark DataFrame (df)
       takes a Delta table path
       takes a checkpoint path

    Writing streaming data to Delta Lake with checkpointing.
    """
    print(f"Writing to Delta Lake: {output_path}")
    print(f"Checkpoint location: {checkpoint_path}")
    

    # df.writeStream = df.writeStream is spark method which is used for writing streaming data again and again as it arrives.
    # basically df.writeStream It defines how streaming data should be written

    # .format("delta") = Specifies that we want to write the data in Delta Lake format (This tells Spark: “Write the data as a Delta Lake table.”)

    # .outputMode("append") = We want to append new data to the existing Delta table (Tells Spark: add new records without overwriting)
    # .outputMode -> tells Spark how the results of a streaming query should be written to the output. it controls whether Spark writes new rows, updated rows, or the entire result each time data arrives.

    # .option("checkpointLocation", checkpoint_path) = Specifies where Spark should store checkpoint information (Tells Spark: save progress and state here for fault tolerance)


    # .option("mergeSchema", "true") = Allows Spark to automatically update the schema if new fields are added (Tells Spark: if the data structure changes, adapt the schema automatically)
    #  .option("mergeSchema", "true") -> Allows new columns to be added in the future Streaming job does not break on schema changes

    # .start(output_path) = Starts the streaming query and specifies where to write the Delta Lake data (Tells Spark: start writing the data to this location)
    # .start() actually starts the writingStreaming query and returns a StreamingQuery object (Streaming DataFrame -> writeStream -> start(output_path)  →  data is written here)

    query = df.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", checkpoint_path) \
        .option("mergeSchema", "true") \
        .start(output_path)
    
    return query


def main():
    """
    Main function to run the Bronze layer ingestion pipeline.
    """
    print("=" * 60)
    print("Bronze Layer - Raw Data Ingestion")
    print("=" * 60)
    print(f"Kafka Bootstrap Servers: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Kafka Topic: {KAFKA_TOPIC}")
    print(f"Bronze Path: {BRONZE_PATH}")
    print(f"Checkpoint Path: {CHECKPOINT_PATH}")
    print("=" * 60)
    
    # Create Spark session
    spark = create_spark_session()
    print("✓ Spark session created")
    
    # Read from Kafka
    kafka_df = read_kafka_stream(spark)
    print("✓ Kafka stream connected")
    
    # Parse events
    parsed_df = parse_events(kafka_df)
    print("✓ Event parsing configured")
    
    # Write to Delta Lake
    query = write_to_delta(parsed_df, BRONZE_PATH, CHECKPOINT_PATH)
    print("✓ Delta Lake writer started")
    
    print("\nStreaming query running... Press Ctrl+C to stop")
    
    try: # Keep the streaming query running until interrupted
        query.awaitTermination() # Waits for the termination of the streaming query
    except KeyboardInterrupt: # Handle Ctrl+C interruption
        print("\nStopping streaming query...")
        query.stop() # Stop the streaming query
        spark.stop() # Stop the Spark session
        print("Bronze layer ingestion stopped.")


if __name__ == "__main__":
    main()
