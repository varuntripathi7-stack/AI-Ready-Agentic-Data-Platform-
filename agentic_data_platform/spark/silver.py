#!/usr/bin/env python3

"""
Silver Layer - Clean and Validated Data
Reads Bronze Delta table, applies data quality rules, and writes cleaned data.
FAILS on invalid data - strict validation enforced.
"""

import os
# Ensure compatible Java version for Spark
if os.path.isdir("/usr/lib/jvm/java-17-openjdk-amd64"):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"




# # SparkSession is the entry point to Spark.
# It starts the Spark engine and lets us read, process, and write data using DataFrames.
from pyspark.sql import SparkSession





# Spark SQL functions for data transformations and validations 
# We will use these functions to check for nulls, filter data, and perform transformations.
# col -> used to reference columns in DataFrames -> (Refer to DataFrame columns safely)
# to_timestamp -> converts string to timestamp type 
# trim -> removes leading/trailing whitespace -> (Remove extra spaces)
# lower -> converts string to lowercase 
# count -> counts the number of records that match a condition. ->(Count records)
# lit -> creates a literal value (used for adding constant columns) -> (Create constant columns (future-proofing))
from pyspark.sql.functions import (
    col, to_timestamp, trim, lower, count, lit
)


# We will also use TimestampType to ensure our event_timestamp column is properly typed in the Silver layer.
from pyspark.sql.types import TimestampType 



import sys 


# Here we are defining the paths for bronze data storage, silver data storage, and Spark checkpointing
# We use os.path to construct these paths in a way that is portable across different operating systems.
# BASE_PATH is the root directory of our project, and we build the other paths relative to it.
# This way, we can easily manage our data and checkpoints without hardcoding absolute paths.
# The checkpoint path is important for Spark streaming jobs, but we include it here for future-proofing in case we want to convert this batch job to a streaming job later on.
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get the base directory of the project
BRONZE_PATH = os.path.join(BASE_PATH, "data/bronze/ecommerce_events") # Path to store raw bronze data
SILVER_PATH = os.path.join(BASE_PATH, "data/silver/ecommerce_events") # Path to store cleaned silver data
CHECKPOINT_PATH = os.path.join(BASE_PATH, "data/checkpoints/silver")  # Path for Spark checkpointing





# here we are defining the event types that are allowed to exist in the Silver layer.
# Valid event types 
VALID_EVENT_TYPES = ["view", "cart", "purchase"]





# This function starts Spark and prepares it to read streaming data from Kafka 
# and write data to Delta Lake, using the required configurations and dependencies.
def create_spark_session() -> SparkSession:
    """
    Create and configure Spark session with Delta Lake support. 
    """
    # Building a Spark session with necessary configurations for Delta Lake and Kafka integration.
    # Note -> We are using Delta Lake to store raw streaming data from Kafka in a 
    # safe, reliable, and replayable way.

    # Configurations include:
    # spark.sql.extensions -> to enable Delta Lake SQL extensions
    # spark.sql.catalog.spark_catalog -> to use Delta Lake as the default catalog
    # spark.jars.packages -> to include the Delta Lake Spark connector dependency
    # We specify the version of Delta Lake that is compatible with our Spark version (3.1.0 in this case).
    # This setup allows us to read and write Delta tables seamlessly in our Spark application.
    # We also set the log level to WARN to reduce verbosity in the console output, making it easier to spot important messages and errors.
    # Finally, we return the created Spark session object for use in our main processing logic.
    
    #Note:-   
    #  .appName -> Sets the name of the Spark application.
    #  .config -> Configures various Spark settings.
    #  .getOrCreate() -> Creates the Spark session if it doesn't exist, otherwise returns the existing one.
    spark = SparkSession.builder \
        .appName("Silver_Layer_Cleaning") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN") # Set log level to WARN to reduce verbosity
    return spark







# This function reads raw data from the Bronze Delta table, verifies that data exists, 
# and safely returns it to the pipeline — or fails the job if something is wrong.

def read_bronze_data(spark: SparkSession): # We pass the Spark session as an argument to this function so that it can use it to read data from the Bronze Delta table.
    """
    Read data from Bronze Delta table.
    """
    print(f"Reading from Bronze path: {BRONZE_PATH}")
    
    try:
        bronze_df = spark.read.format("delta").load(BRONZE_PATH) # Read the Bronze Delta table into a DataFrame
        record_count = bronze_df.count() # Count the number of records read from the Bronze layer to provide feedback on the volume of data being processed.
        print(f"✓ Read {record_count} records from Bronze layer")  # Print a success message with the number of records read from the Bronze layer.
        return bronze_df # Return the DataFrame containing the Bronze data for further processing in the pipeline.
    except Exception as e: 
        print(f"✗ Failed to read Bronze data: {e}")
        sys.exit(1) # Exit the program with a non-zero status code to indicate failure if there was an error reading the Bronze data.






#This function checks whether the Bronze data is clean enough to be trusted, and 
# stops the entire pipeline if any serious data quality problem is found.
def validate_data_quality(df, spark: SparkSession) -> bool:
    """
    Check for data quality issues that should cause failure.
    Returns True if data is valid, False otherwise.
    """
    print("\nRunning data quality validations...")
    
    issues = [] # List to collect data quality issues found
    
    # Check 1: Null user_id
    null_users = df.filter(col("user_id").isNull()).count()
    if null_users > 0:
        issues.append(f"Found {null_users} records with NULL user_id")
    
    # Check 2: Invalid event_type
    invalid_events = df.filter(
        ~lower(trim(col("event_type"))).isin([e.lower() for e in VALID_EVENT_TYPES]) # Normalize event_type to lowercase and check if it's in valid types
    ).count() # Count records with invalid event_type
    if invalid_events > 0:
        issues.append(f"Found {invalid_events} records with invalid event_type")
    
    # Check 3: Null product_id
    null_products = df.filter(col("product_id").isNull()).count()
    if null_products > 0:
        issues.append(f"Found {null_products} records with NULL product_id")
    
    # Check 4: Negative prices
    negative_prices = df.filter(col("price") < 0).count()
    if negative_prices > 0:
        issues.append(f"Found {negative_prices} records with negative price")
    
    # Check 5: Purchase events with zero price
    zero_price_purchases = df.filter(
        (lower(trim(col("event_type"))) == "purchase") & 
        ((col("price").isNull()) | (col("price") <= 0))
    ).count()
    if zero_price_purchases > 0:
        issues.append(f"Found {zero_price_purchases} purchase events with zero/null price")
    
    if issues:
        print("\n" + "=" * 60)
        print("DATA QUALITY ISSUES DETECTED - JOB WILL FAIL")
        print("=" * 60)
        for issue in issues: # Print each data quality issue found
            print(f"  ✗ {issue}")
        print("=" * 60)
        return False # Return False to indicate that data quality validation failed
    
    print("✓ All data quality checks passed")
    return True # Return True to indicate that data quality validation passed





# This function applies cleaning transformations to the Bronze data to prepare it for the Silver layer.

#This function takes already-validated Bronze data, cleans and standardizes it, 
# and shapes it into the final Silver-layer schema.
def clean_data(df): # We pass the DataFrame containing the Bronze data to this function, and it will return a new DataFrame that has been cleaned and transformed according to our Silver layer requirements.
    """
    Apply cleaning transformations to the data.
    """
    print("\nApplying data cleaning transformations...")
    
    # Remove duplicates based on user_id, product_id, event_type, timestamp
    cleaned_df = df.dropDuplicates(["user_id", "product_id", "event_type", "event_timestamp"])
    
    # Remove null user_id records
    cleaned_df = cleaned_df.filter(col("user_id").isNotNull())
    
    # Normalize event_type to lowercase
    # df.withColumn() -> is a spark function which is used to create a new column or modify an existing column in a Spark DataFrame.
    # df.withColumn() -> (Think of it as: “Add or update a column.”)
    cleaned_df = cleaned_df.withColumn("event_type", lower(trim(col("event_type"))))
    
    # Filter only valid event types
    cleaned_df = cleaned_df.filter(col("event_type").isin(VALID_EVENT_TYPES))
    
    # Convert timestamp string to proper timestamp type
    cleaned_df = cleaned_df.withColumn(
        "event_timestamp_parsed",
        to_timestamp(col("event_timestamp"))
    )
    
    # Remove invalid timestamps
    cleaned_df = cleaned_df.filter(col("event_timestamp_parsed").isNotNull())
    
    # Select and rename columns for Silver layer
    # here we are basically performing a SQL SELECT operation.
    silver_df = cleaned_df.select(
        col("user_id"),
        col("product_id"),
        col("event_type"),
        col("price"),
        col("event_timestamp_parsed").alias("event_timestamp"),
        col("kafka_timestamp").alias("ingestion_timestamp")
    )
    
    return silver_df # Return the cleaned DataFrame that is now ready to be written to the Silver layer.




# This function takes the cleaned Silver-ready data and safely writes it to a Delta table, 
# then reports how many records were written.
def write_to_silver(df, output_path: str): 
    """
    We pass the cleaned DataFrame and the output path for the Silver Delta table to this function, 
    which will handle writing the data to storage and return the count of records written.
    
    This function takes:

        df -> the cleaned Silver DataFrame

        output_path -> where to save the Silver table

    Write cleaned data to Silver Delta table.
    """
    print(f"\nWriting to Silver path: {output_path}")
    
    record_count = df.count() # Count the number of records in the cleaned DataFrame to provide feedback on how many records are being written to the Silver layer.
    
    """ 
    # df.write -> is a Spark DataFrameWriter object that allows us to specify how we want to write the DataFrame to storage.

    # .format("delta") -> specifies that we want to write the data in Delta Lake format.

    # .mode("overwrite") -> specifies that if there is already data at the output path, 
       it should be overwritten with the new data. This is important for our Silver layer because 
       we want to ensure that it always reflects the latest cleaned data from the Bronze layer.


    # .option("overwriteSchema", "true") -> allows the schema of the Delta table to be overwritten 
      if it has changed. This is useful for future-proofing our pipeline, as it allows us to make changes 
      to the data schema in the future without having to manually manage schema changes in the Delta table.

      
    # .save(output_path) -> specifies the path where the Silver Delta table should be saved. 
       This will create a new Delta table at the specified location with the cleaned data from the DataFrame.
    
    """
    df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .save(output_path)
    
    print(f"✓ Wrote {record_count} records to Silver layer")
    return record_count





# This main() function orchestrates the full Silver layer workflow: (Orchestrates means controlling and 
# coordinating different steps so they run in the right order.)
# start Spark → read Bronze → validate → clean → write Silver → shut down Spark.
def main():
    """
    Main function to run the Silver layer processing pipeline.
    """
    print("=" * 60)
    print("Silver Layer - Data Cleaning & Validation")
    print("=" * 60)
    print(f"Bronze Path: {BRONZE_PATH}")
    print(f"Silver Path: {SILVER_PATH}")
    print("=" * 60)
    
    # Create Spark session
    spark = create_spark_session()
    print("✓ Spark session created")
    
    # Read Bronze data
    bronze_df = read_bronze_data(spark)
    



    # Validate data quality - FAIL if issues found
    """
    We pass the Bronze DataFrame and the Spark session to the validate_data_quality function, which will 
    check for data quality issues and return True if the data is valid or False if there are issues 
    that should cause the pipeline to fail.
    """
    is_valid = validate_data_quality(bronze_df, spark) 
    
    if not is_valid:
        print("\n✗ PIPELINE FAILED: Data quality validation errors detected!")
        print("Please investigate and fix the data quality issues in Bronze layer.")
        spark.stop()
        sys.exit(1)
    
    # Clean data
    silver_df = clean_data(bronze_df)
    
    # Write to Silver
    records_written = write_to_silver(silver_df, SILVER_PATH)
    
    print("\n" + "=" * 60)
    print("Silver Layer Processing Complete")
    print(f"Records processed: {records_written}")
    print("=" * 60)
    
    spark.stop() # Stop the Spark session to free up resources after the processing is complete.


if __name__ == "__main__":
    main()
