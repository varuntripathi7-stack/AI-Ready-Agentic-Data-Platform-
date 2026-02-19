#!/usr/bin/env python3
"""
Gold Layer - Business Aggregations
Creates business-ready aggregated tables from Silver data.
Outputs:
  - Revenue per hour
  - Active users per hour
  - Conversion rate
"""

import os
# Ensure compatible Java version for Spark
if os.path.isdir("/usr/lib/jvm/java-17-openjdk-amd64"):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"



# Import SparkSession to create a Spark session for processing data.
from pyspark.sql import SparkSession 



# Import Spark SQL functions for data transformations and aggregations.
# These functions allow us to manipulate and analyze the data in various ways, such as filtering, grouping, and calculating metrics.

# col: Used to reference a column in a DataFrame.
# hour: Extracts the hour from a timestamp.
# date_trunc: Truncates a timestamp to a specified unit (e.g., hour).
# sum: Calculates the sum of a column.
# count: Counts the number of rows or non-null values in a column.
# countDistinct: Counts the number of distinct values in a column.
# when: Used for conditional expressions in DataFrame transformations.
# round: Rounds a numeric column to a specified number of decimal places.
# lit: Creates a column with a literal value.
from pyspark.sql.functions import (
    col, hour, date_trunc, sum as spark_sum, count, countDistinct,
    when, round as spark_round, lit
)




# Import sys for handling system-level operations, such as exiting the program in case of errors.
import sys




# Configuration

# Define file paths for Silver input and Gold output.
# These paths are constructed based on the directory structure of the project.
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get the base directory of the project
SILVER_PATH = os.path.join(BASE_PATH, "data/silver/ecommerce_events") # Path to the Silver Delta table
GOLD_PATH = os.path.join(BASE_PATH, "data/gold") # Define the path for the Gold layer output
REVENUE_PATH = os.path.join(GOLD_PATH, "revenue_per_hour") # Define the path for the revenue per hour output
ACTIVE_USERS_PATH = os.path.join(GOLD_PATH, "active_users_per_hour") # Define the path for the active users per hour output
CONVERSION_RATE_PATH = os.path.join(GOLD_PATH, "conversion_rate") # Define the path for the conversion rate output



# This function creates and configures a Spark session with Delta Lake support, which is necessary for 
# reading and writing Delta tables in the Silver and Gold layers.
def create_spark_session() -> SparkSession: 
    """
    Create and configure Spark session with Delta Lake support.


    This function initializes a Spark session with the necessary configurations to work with Delta Lake.

    .appName -> Sets the name of the Spark application, which will be displayed in the Spark UI.

    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") -> Adds the Delta Lake extension to Spark, enabling support for Delta Lake features.

    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") -> Configures the Spark catalog to use Delta Lake.

    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") -> Specifies the Delta Lake package to be used.
    """
    spark = SparkSession.builder \
        .appName("Gold_Layer_Aggregations") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN") # Set log level to WARN to reduce verbosity in the console output.
    return spark # Return the created Spark session to be used for data processing in the Gold layer.





# This function safely loads Silver data and makes sure the pipeline stops if the data is missing or broken
def read_silver_data(spark: SparkSession): 
    """
    Read data from Silver Delta table.
    """
    print(f"Reading from Silver path: {SILVER_PATH}")
    
    try:
        silver_df = spark.read.format("delta").load(SILVER_PATH) # Load the Silver Delta table into a DataFrame. 
        record_count = silver_df.count() # Count the number of records in the DataFrame to confirm successful loading. 
        print(f"✓ Read {record_count} records from Silver layer") 
        return silver_df
    except Exception as e:
        print(f"✗ Failed to read Silver data: {e}")
        sys.exit(1) # Exit the program with a non-zero status code to indicate failure.





# These functions perform the actual business aggregations on the Silver data to create the Gold tables.
# Each function takes the Silver DataFrame and the Spark session as input, performs the necessary transformations and aggregations, and returns a new DataFrame with the results.
def calculate_revenue_per_hour(df, spark: SparkSession):
    """
    Calculate total revenue aggregated by hour.
    Only includes purchase events.
    """
    print("\nCalculating Revenue per Hour...")
    
    # Filter for purchase events and aggregate by hour

    # .filter(col("event_type") == "purchase") -> Filter purchase events

    # .withColumn("hour", date_trunc("hour", col("event_timestamp"))) -> Create a new column "hour" by truncating the event timestamp to the hour level.

    # .groupBy("hour") -> Group the data by the "hour" column to perform aggregations for each hour.

    # .agg(...) -> Perform multiple aggregations:
    #   - spark_round(spark_sum("price"), 2).alias("total_revenue") -> Calculate the total revenue by summing the "price" column and rounding it to 2 decimal places.
    #   - count("*").alias("purchase_count") -> Count the number of purchase events for each hour.
    #   - spark_round(spark_sum("price") / count("*"), 2).alias("avg_order_value") -> Calculate the average order value by dividing the total revenue by the purchase count and rounding it to 2 decimal places.
    # .orderBy(col("hour").desc()) -> Order the results by hour in descending order to show the most recent hours first.    
    revenue_df = df \
        .filter(col("event_type") == "purchase") \
        .withColumn("hour", date_trunc("hour", col("event_timestamp"))) \
        .groupBy("hour") \
        .agg(
            spark_round(spark_sum("price"), 2).alias("total_revenue"),
            count("*").alias("purchase_count"),
            spark_round(spark_sum("price") / count("*"), 2).alias("avg_order_value")
        ) \
        .orderBy(col("hour").desc())
    
    # Show sample output
    print("\nSample Revenue per Hour:")
    revenue_df.show(5, truncate=False) # Display the first 5 rows of the revenue DataFrame to verify the results of the aggregation.
    
    return revenue_df # Return the resulting DataFrame containing the revenue per hour to be written to the Gold layer later in the pipeline.





def calculate_active_users_per_hour(df, spark: SparkSession):
    """
    Calculate distinct active users aggregated by hour.
    Counts unique users who performed any action.
    """
    print("\nCalculating Active Users per Hour...")
    
    # Aggregate distinct users by hour
    active_users_df = df \
        .withColumn("hour", date_trunc("hour", col("event_timestamp"))) \
        .groupBy("hour") \
        .agg(
            countDistinct("user_id").alias("active_users"),
            count("*").alias("total_events"),
            countDistinct("product_id").alias("unique_products_viewed")
        ) \
        .orderBy(col("hour").desc())
    
    # Show sample output
    print("\nSample Active Users per Hour:")
    active_users_df.show(5, truncate=False)
    
    return active_users_df






def calculate_conversion_rate(df, spark: SparkSession):
    """
    Calculate conversion rate (views -> cart -> purchase funnel).
    Conversion rate = purchases / views * 100
    """
    print("\nCalculating Conversion Rate...")
    
    # Count by event type per hour
    conversion_df = df \
        .withColumn("hour", date_trunc("hour", col("event_timestamp"))) \
        .groupBy("hour") \
        .agg(
            count(when(col("event_type") == "view", 1)).alias("view_count"),
            count(when(col("event_type") == "cart", 1)).alias("cart_count"),
            count(when(col("event_type") == "purchase", 1)).alias("purchase_count"),
            countDistinct("user_id").alias("unique_users")
        ) \
        .withColumn(
            "view_to_cart_rate",
            spark_round(
                when(col("view_count") > 0, col("cart_count") / col("view_count") * 100)
                .otherwise(0.0), 2
            )
        ) \
        .withColumn(
            "cart_to_purchase_rate",
            spark_round(
                when(col("cart_count") > 0, col("purchase_count") / col("cart_count") * 100)
                .otherwise(0.0), 2
            )
        ) \
        .withColumn(
            "overall_conversion_rate",
            spark_round(
                when(col("view_count") > 0, col("purchase_count") / col("view_count") * 100)
                .otherwise(0.0), 2
            )
        ) \
        .orderBy(col("hour").desc())
    
    # Show sample output
    print("\nSample Conversion Rates:")
    conversion_df.select(
        "hour", "view_count", "cart_count", "purchase_count",
        "view_to_cart_rate", "cart_to_purchase_rate", "overall_conversion_rate"
    ).show(5, truncate=False)
    
    return conversion_df






def write_to_delta(df, output_path: str, table_name: str):
    """
    Write aggregated data to Delta Lake.
    """
    print(f"Writing {table_name} to: {output_path}")
    
    df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .save(output_path)
    
    record_count = df.count()
    print(f"✓ Wrote {record_count} records to {table_name}")


def main():
    """
    Main function to run the Gold layer aggregation pipeline.
    """
    print("=" * 60)
    print("Gold Layer - Business Aggregations")
    print("=" * 60)
    print(f"Silver Path: {SILVER_PATH}")
    print(f"Gold Path: {GOLD_PATH}")
    print("=" * 60)
    
    # Create Spark session
    spark = create_spark_session()
    print("✓ Spark session created")
    
    # Read Silver data
    silver_df = read_silver_data(spark)
    
    # Cache for multiple aggregations
    silver_df.cache()
    
    # Calculate aggregations
    revenue_df = calculate_revenue_per_hour(silver_df, spark)
    active_users_df = calculate_active_users_per_hour(silver_df, spark)
    conversion_df = calculate_conversion_rate(silver_df, spark)
    
    # Write to Gold layer
    print("\n" + "=" * 60)
    print("Writing Gold Tables")
    print("=" * 60)
    
    write_to_delta(revenue_df, REVENUE_PATH, "revenue_per_hour")
    write_to_delta(active_users_df, ACTIVE_USERS_PATH, "active_users_per_hour")
    write_to_delta(conversion_df, CONVERSION_RATE_PATH, "conversion_rate")
    
    # Unpersist cached data
    silver_df.unpersist()
    
    print("\n" + "=" * 60)
    print("Gold Layer Processing Complete")
    print("=" * 60)
    print(f"  - Revenue per hour: {REVENUE_PATH}")
    print(f"  - Active users per hour: {ACTIVE_USERS_PATH}")
    print(f"  - Conversion rate: {CONVERSION_RATE_PATH}")
    print("=" * 60)
    
    spark.stop()


if __name__ == "__main__":
    main()
