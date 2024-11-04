import pandas as pd
import time
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Set the Kafka bootstrap server and topic name
bootstrap_servers = "localhost:9092"
topic_name = "gait-test"

# Load and shuffle test dataset
test_file_path = "/home/ayemon/KafkaProjects/Dataset/gait_data/feature_80_20/test_all.csv"
test_df = pd.read_csv(test_file_path)
test_df = test_df.sample(frac=1).reset_index(drop=True)

# Encode the surface_type column
surface_type_mapping = {
    "FLAT": 0,
    "GRAVEL": 2,
    "GRASS": 1
}
test_df["surface_type"] = test_df["surface_type"].map(surface_type_mapping)

# Initialize the Kafka producer
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

# Error callback function
def error_callback(exc):
    raise Exception('Error while sending data to Kafka: {0}'.format(str(exc)))

# Function to send data to Kafka with a 5-second break after every 320 records
def write_to_kafka(topic_name, items):
    count = 0
    for message, key in items:
        producer.send(topic_name, key=key.encode('utf-8'), value=message.encode('utf-8')).add_errback(error_callback)
        count += 1
        print(f"Message sent: {message}")
        print(f"Key: {key}")

        # Take a 5-second break after sending every 320 messages
        if count > 5000 and count% 640 == 0:
            print("Taking a 30-second break...")
            time.sleep(60)

    producer.flush()
    print(f"Wrote {count} messages into topic: {topic_name}")

# Prepare the data
x_test_df = test_df.drop(["surface_type"], axis=1)
y_test_df = test_df["surface_type"]
x_test = list(filter(None, x_test_df.to_csv(index=False, header=False).split("\n")[1:]))
y_test = list(filter(None, y_test_df.to_csv(index=False, header=False).split("\n")[1:]))

# Send the data to Kafka
write_to_kafka(topic_name, zip(x_test, y_test))

# Close the producer connection
producer.close()
