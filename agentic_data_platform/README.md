# Agentic Data Platform

This directory contains the core application code.  
See the [main README](../README.md) for full architecture, setup instructions, and documentation.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Kafka
docker-compose up -d

# 3. Run the pipeline
python simulator/event_generator.py       # Terminal 1 - generate events
spark-submit --packages io.delta:delta-spark_2.12:3.1.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 spark/bronze.py  # Terminal 2
spark-submit --packages io.delta:delta-spark_2.12:3.1.0 spark/silver.py
spark-submit --packages io.delta:delta-spark_2.12:3.1.0 spark/gold.py
spark-submit --packages io.delta:delta-spark_2.12:3.1.0 features/build_features.py
python ml/train_model.py

# 4. Start the AI agent
ollama pull mistral && ollama serve        # Terminal 3
python agent/agent.py                      # Terminal 4
```
