import os
import pandas as pd
import json
import paho.mqtt.client as mqtt
import torch  # Import PyTorch
from torch import nn  # For defining neural networks
import torch.optim as optim  # For optimizers
from kafka import KafkaConsumer
import json
from river import metrics, preprocessing, compose
from deep_river.classification import Classifier
import matplotlib.pyplot as plt
import time

# The rest of the code remains unchanged...

# Configuration
folder_path = "/home/ayemon/KafkaProjects/Dataset/gait_data/123456/123456/"  # Replace with your folder path
mqtt_broker = "10.87.44.24"  # Replace with your broker's IP
mqtt_port = 1883  # Port
mqtt_topic = "temp-gait"  # Topic
username = "ayemon12345"  # MQTT username
password = "abc12345"  # MQTT password
batch_size = 30

# Columns to send (excluding Timestamp (s))
columns_to_send = [
    "Gyroscope X (deg/s)", "Gyroscope Y (deg/s)", "Gyroscope Z (deg/s)",
    "Accelerometer X (g)", "Accelerometer Y (g)", "Accelerometer Z (g)", "Surface_type"
]


# MQTT connection callback
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")


# Setup MQTT client
client = mqtt.Client()
client.username_pw_set(username, password)
client.on_connect = on_connect
client.connect(mqtt_broker, mqtt_port, 60)


def process_files(folder_path, batch_size):
    # Get all CSV files in the folder
    all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    total_files = len(all_files)

    # Initialize per-file row counters
    row_counters = [0] * total_files  # Tracks the current position in each file

    while True:
        has_data = False  # Track if there's data left to process

        for idx, file_path in enumerate(all_files):
            # Read the CSV file
            df = pd.read_csv(file_path)
            df = df[columns_to_send]  # Keep only required columns

            start_row = row_counters[idx]
            end_row = start_row + batch_size

            # If there's data left in the file
            if start_row < len(df):
                has_data = True
                data_to_send = df.iloc[start_row:end_row].to_dict(orient="records")

                # Publish the batch to MQTT
                payload = {
                    "file_name": os.path.basename(file_path),
                    "batch_start": start_row + 1,
                    "batch_end": min(end_row, len(df)),
                    "data": data_to_send
                }
                client.publish(mqtt_topic, json.dumps(payload))
                print(f"Published rows {start_row + 1}-{end_row} from {os.path.basename(file_path)}")

                # Update the row counter for the file
                row_counters[idx] += batch_size

        if not has_data:
            print("All files processed.")
            break  # Exit the loop if no data is left to process in any file

if __name__ == "__main__":
    client.loop_start()
    try:
        process_files(folder_path, batch_size)
    finally:
        client.loop_stop()
        client.disconnect()
