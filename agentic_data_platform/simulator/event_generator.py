#!/usr/bin/env python3
"""
Event Generator for E-commerce Data Platform
Generates random e-commerce events and sends them to Kafka.
"""

import json 
import random 
import time 
import sys 
from datetime import datetime, timezone 
from kafka import KafkaProducer  # KafkaProducer accepts python data (python object)->(stores in byte) and send it to kafka
from kafka.errors import KafkaError # KafkaError is the base class used for handling Kafka-related failures

# Note -> Kafka is like a post office - it receives messages and delivers them.

# here we are doing Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "ecommerce_events"


# here we are doing Event Configuration
EVENT_TYPES = ["view", "cart", "purchase"]
EVENT_WEIGHTS = [0.7, 0.2, 0.1]      # 70% views, 20% carts, 10% purchases


# here we Define the range for user IDs and product IDs
USER_ID_RANGE = (1, 10000)
PRODUCT_ID_RANGE = (1, 5000)

# here we Define the price range for purchase events
PRICE_MIN = 5.0
PRICE_MAX = 500.0


def generate_event() -> dict: # This function generates one random e-commerce event and returns it as a Python dictionary.
    """
    Generate a single random e-commerce event.
    
    Returns:
        dict: Event following the schema:
            - user_id: int
            - product_id: int
            - event_type: str (view|cart|purchase)
            - price: float (0.0 for view/cart, >0 for purchase)
            - timestamp: str (ISO-8601 format)
    """

   # Select a single event type using weighted random selection and extract the value from the returned list
    event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0]

    # Generate user and product IDs
    user_id = random.randint(*USER_ID_RANGE)
    product_id = random.randint(*PRODUCT_ID_RANGE)


   # BUSINESS LOGIC:- 
   # If the event is a purchase, generate a random product price; otherwise set price to 0 for non-revenue events
   # Price is 0 for view/cart
    if event_type == "purchase":
        price = round(random.uniform(PRICE_MIN, PRICE_MAX), 2)
    else:
        price = 0.0  # Price is 0 for view/cart
    
  
   # Generate ISO-8601 timestamp
    timestamp = datetime.now(timezone.utc).isoformat()


    event = {
        "user_id": user_id,
        "product_id": product_id,
        "event_type": event_type,
        "price": price,
        "timestamp": timestamp
    }
    
    return event



def create_kafka_producer() -> KafkaProducer:
    """
    Create and return a Kafka producer instance.
    
    Returns:
        KafkaProducer: Configured Kafka producer
    """
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),  # convert value to bytes (JSON → UTF-8)
            key_serializer=lambda k: k.encode('utf-8') if k else None, # convert key to bytes (string → UTF-8)
            acks='all',  # Wait for all replicas to acknowledge
            retries=4,   # Retry failed sends, (If sending a message fails, KafkaProducer will retry sending it up to 4 times before giving up.)
            retry_backoff_ms=500 # After a failure: wait 500 milliseconds then retry
        )
        print(f"✓ Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
        return producer
    except KafkaError as e:
        print(f"✗ Failed to connect to Kafka: {e}")
        sys.exit(1) # Stop the program immediately and indicate that an error occurred


  

    
def on_send_success(record_metadata):   # This function is called automatically after a message is sent successfully; no action is needed on success
    """Callback for successful message delivery."""
    pass  # Silent success





def on_send_error(excp): # This function is called automatically after a message is sent failed and print failed message 
    """Callback for failed message delivery."""
    print(f"✗ Failed to send message: {excp}")

  




def main():
    """
    This main() function initializes the Kafka producer, continuously generates and sends events in real time, 
    handles asynchronous delivery callbacks, and ensures a graceful shutdown on user interruption.
    
    Main function to generate events and send them to Kafka.
    Generates 1 event per second, runs infinitely.
    """
    
    print("=" * 60)
    print("E-commerce Event Generator (Kafka Mode)")
    print(f"Topic: {KAFKA_TOPIC}")
    print(f"Bootstrap Servers: {KAFKA_BOOTSTRAP_SERVERS}")
    print("Generating 1 event per second...")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Create Kafka producer
    producer = create_kafka_producer()
    
    event_count = 0
    
    try:
        while True:
            # Generate event
            event = generate_event()
            event_count += 1
            
            # Use user_id as partition key for ordering
            key = str(event["user_id"])
            
            # Send to Kafka asynchronously
            future = producer.send(
                KAFKA_TOPIC,
                key=key,
                value=event
            )
            
            # Add callbacks
            future.add_callback(on_send_success)
            future.add_errback(on_send_error)
            
            # Print event info
            print(f"[Event #{event_count}] Sent: {event['event_type']} | "
                  f"user: {event['user_id']} | "
                  f"product: {event['product_id']} | "
                  f"price: ${event['price']:.2f}")
            
            # Wait 1 second before next event
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n\nShutting down...")
        producer.flush()  # Ensure all messages are sent
        producer.close()
        print(f"Generator stopped. Total events sent: {event_count}")


if __name__ == "__main__":
    main()
