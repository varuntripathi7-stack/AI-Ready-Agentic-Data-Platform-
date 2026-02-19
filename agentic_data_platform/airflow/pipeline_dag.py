#!/usr/bin/env python3
"""
Airflow DAG for E-commerce Data Platform Pipeline
Orchestrates the complete data pipeline from Silver through ML training.

Schedule:
  - Silver Layer: Every 15 minutes
  - Gold Layer: Every 15 minutes (after Silver)
  - Feature Engineering: Hourly
  - ML Training: Daily
"""

from datetime import datetime, timedelta
import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.task.trigger_rule import TriggerRule

# Configuration
PROJECT_PATH = "/opt/airflow/dags/agentic_data_platform"
SPARK_HOME = "/opt/spark"
PYTHON_PATH = "/usr/bin/python3"

SPARK_PACKAGES = "io.delta:delta-spark_2.12:3.1.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"

# Default arguments for all tasks
default_args = {
    'owner': 'data-platform',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}


# ============================================================
# Main Pipeline DAG - Runs every 15 minutes
# ============================================================
with DAG(
    dag_id='ecommerce_data_pipeline',
    default_args=default_args,
    description='E-commerce data pipeline: Silver -> Gold layers',
    schedule='*/15 * * * *',  # Every 15 minutes
    start_date=pendulum.today('UTC').add(days=-1),
    catchup=False,
    max_active_runs=1,
    tags=['ecommerce', 'delta-lake', 'spark'],
) as data_pipeline_dag:
    
    # Start marker
    start = EmptyOperator(task_id='start')
    
    # Silver Layer Processing
    silver_layer = BashOperator(
        task_id='silver_layer_processing',
        bash_command=f"""
            {SPARK_HOME}/bin/spark-submit \
                --packages {SPARK_PACKAGES} \
                --driver-memory 2g \
                --executor-memory 2g \
                {PROJECT_PATH}/spark/silver.py
        """,
        retries=3,
        retry_delay=timedelta(minutes=2),
    )
    
    # Gold Layer Processing
    gold_layer = BashOperator(
        task_id='gold_layer_processing',
        bash_command=f"""
            {SPARK_HOME}/bin/spark-submit \
                --packages {SPARK_PACKAGES} \
                --driver-memory 2g \
                --executor-memory 2g \
                {PROJECT_PATH}/spark/gold.py
        """,
        retries=2,
    )
    
    # Data quality check
    def check_gold_data_quality(**context):
        """Check if Gold layer data is valid."""
        import os
        import glob
        
        project_path = "/opt/airflow/dags/agentic_data_platform"
        gold_paths = [
            f"{project_path}/data/gold/revenue_per_hour",
            f"{project_path}/data/gold/active_users_per_hour",
            f"{project_path}/data/gold/conversion_rate"
        ]
        
        for path in gold_paths:
            parquet_files = glob.glob(f"{path}/*.parquet")
            if not parquet_files:
                raise ValueError(f"No data found in {path}")
        
        print("✓ All Gold tables have data")
        return True
    
    data_quality_check = PythonOperator(
        task_id='data_quality_check',
        python_callable=check_gold_data_quality,
    )
    
    # End marker
    end = EmptyOperator(
        task_id='end',
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )
    
    # Define task dependencies
    start >> silver_layer >> gold_layer >> data_quality_check >> end


# ============================================================
# Feature Engineering DAG - Runs hourly
# ============================================================
with DAG(
    dag_id='feature_engineering_pipeline',
    default_args=default_args,
    description='Feature engineering pipeline for ML',
    schedule='0 * * * *',  # Every hour
    start_date=pendulum.today('UTC').add(days=-1),
    catchup=False,
    max_active_runs=1,
    tags=['ecommerce', 'features', 'ml'],
) as feature_dag:
    
    start_features = EmptyOperator(task_id='start')
    
    # Wait for data pipeline to complete (sensor could be added here)
    wait_for_data = EmptyOperator(task_id='wait_for_data')
    
    # Build features using Spark
    build_features = BashOperator(
        task_id='build_features',
        bash_command=f"""
            {SPARK_HOME}/bin/spark-submit \
                --packages {SPARK_PACKAGES} \
                --driver-memory 2g \
                --executor-memory 2g \
                {PROJECT_PATH}/features/build_features.py
        """,
        retries=2,
    )
    
    # Validate features
    def validate_features(**context):
        """Validate feature table exists and has data."""
        import glob
        
        project_path = "/opt/airflow/dags/agentic_data_platform"
        features_path = f"{project_path}/data/features/user_features"
        parquet_files = glob.glob(f"{features_path}/*.parquet")
        
        if not parquet_files:
            raise ValueError("Feature table is empty or missing")
        
        print(f"✓ Feature table has {len(parquet_files)} parquet files")
        return True
    
    validate_features_task = PythonOperator(
        task_id='validate_features',
        python_callable=validate_features,
    )
    
    end_features = EmptyOperator(task_id='end')
    
    start_features >> wait_for_data >> build_features >> validate_features_task >> end_features


# ============================================================
# ML Training DAG - Runs daily
# ============================================================
with DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='ML model training pipeline',
    schedule='0 2 * * *',  # Daily at 2 AM
    start_date=pendulum.today('UTC').add(days=-1),
    catchup=False,
    max_active_runs=1,
    tags=['ecommerce', 'ml', 'training'],
) as ml_dag:
    
    start_ml = EmptyOperator(task_id='start')
    
    # Check if we have enough data for training
    def check_training_data(**context):
        """Check if we have sufficient data for training."""
        import glob
        import pandas as pd
        
        project_path = "/opt/airflow/dags/agentic_data_platform"
        features_path = f"{project_path}/data/features/user_features"
        parquet_files = glob.glob(f"{features_path}/*.parquet")
        
        if not parquet_files:
            print("No feature data available, skipping training")
            return 'skip_training'
        
        # Load and check data size
        df = pd.concat([pd.read_parquet(f) for f in parquet_files])
        
        if len(df) < 100:
            print(f"Only {len(df)} samples, need at least 100 for training")
            return 'skip_training'
        
        print(f"✓ {len(df)} samples available for training")
        return 'train_model'
    
    check_data = BranchPythonOperator(
        task_id='check_training_data',
        python_callable=check_training_data,
    )
    
    # Train model
    train_model = BashOperator(
        task_id='train_model',
        bash_command=f"{PYTHON_PATH} {PROJECT_PATH}/ml/train_model.py",
        retries=1,
    )
    
    # Skip training branch
    skip_training = EmptyOperator(task_id='skip_training')
    
    # Validate model
    def validate_model(**context):
        """Validate trained model exists and has good metrics."""
        import json
        import os
        
        project_path = "/opt/airflow/dags/agentic_data_platform"
        metrics_file = f"{project_path}/data/models/metrics.json"
        
        if not os.path.exists(metrics_file):
            raise ValueError("Model metrics file not found")
        
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # Check minimum performance thresholds
        if metrics.get('accuracy', 0) < 0.5:
            raise ValueError(f"Model accuracy too low: {metrics.get('accuracy')}")
        
        print(f"✓ Model validated - Accuracy: {metrics.get('accuracy', 0):.4f}")
        return True
    
    validate_model_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
    )
    
    end_ml = EmptyOperator(
        task_id='end',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    
    # Dependencies
    start_ml >> check_data
    check_data >> train_model >> validate_model_task >> end_ml
    check_data >> skip_training >> end_ml


# ============================================================
# Full Pipeline DAG - Master orchestration (optional)
# ============================================================
with DAG(
    dag_id='full_pipeline_orchestration',
    default_args=default_args,
    description='Full end-to-end pipeline orchestration',
    schedule='0 3 * * *',  # Daily at 3 AM
    start_date=pendulum.today('UTC').add(days=-1),
    catchup=False,
    max_active_runs=1,
    tags=['ecommerce', 'orchestration', 'master'],
) as master_dag:

    start_master = EmptyOperator(task_id='start')
    
    # Trigger data pipeline
    trigger_data_pipeline = TriggerDagRunOperator(
        task_id='trigger_data_pipeline',
        trigger_dag_id='ecommerce_data_pipeline',
        wait_for_completion=True,
        poke_interval=60,
    )
    
    # Trigger feature engineering
    trigger_features = TriggerDagRunOperator(
        task_id='trigger_feature_engineering',
        trigger_dag_id='feature_engineering_pipeline',
        wait_for_completion=True,
        poke_interval=60,
    )
    
    # Trigger ML training
    trigger_ml = TriggerDagRunOperator(
        task_id='trigger_ml_training',
        trigger_dag_id='ml_training_pipeline',
        wait_for_completion=True,
        poke_interval=60,
    )
    
    end_master = EmptyOperator(task_id='end')
    
    start_master >> trigger_data_pipeline >> trigger_features >> trigger_ml >> end_master
