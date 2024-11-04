import pandas as pd
import time
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Kafka configuration
bootstrap_servers = "localhost:9092"  # Update with your Kafka server address
topic_name = "river-train"  # Kafka topic for the training data

# File path to the training dataset
train_file_path = "/home/ayemon/KafkaProjects/Dataset/gait_data/feature_80_20/train_all.csv"

# Load and shuffle the training dataset
train_df = pd.read_csv(train_file_path)
train_df = train_df.sample(frac=1).reset_index(drop=True)  # Shuffle the training data

# Initialize the Kafka producer
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

# Error callback function
def error_callback(exc):
    print(f"Error while sending data to Kafka: {str(exc)}")

# Function to send data to Kafka
def write_to_kafka(topic_name, items):
    count = 0
    for message, key in items:
        try:
            # Sending message to Kafka topic
            producer.send(
                topic_name,
                key=key.encode('utf-8'),
                value=message.encode('utf-8')
            ).add_errback(error_callback)

            count += 1
            print(f"Message sent: {message}")
            print(f"Key: {key}")

        except KafkaError as e:
            print(f"Kafka error: {e}")

    producer.flush()
    print(f"Wrote {count} messages into topic: {topic_name}")

# Prepare the data
y_train_df = train_df["surface_type"]
x_train_df = train_df.drop("surface_type", axis=1)
x_train = list(filter(None, x_train_df.to_csv(index=False, header=False).split("\n")[1:]))
y_train = list(filter(None, y_train_df.to_csv(index=False, header=False).split("\n")[1:]))

# Send the data to Kafka
write_to_kafka(topic_name, zip(x_train, y_train))

# Close the producer connection
producer.close()
