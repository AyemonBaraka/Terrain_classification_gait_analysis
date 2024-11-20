import os
import pandas as pd
import json
import random
import paho.mqtt.client as mqtt

# Configuration
folder_path = "/home/ayemon/KafkaProjects/Dataset/gait_data/123456/123456/"  # Replace with your folder path
mqtt_broker = "10.87.44.24"  # Replace with your broker's IP
mqtt_port = 1883  # Port
mqtt_topic = "gait-missing"  # Topic
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

# Function to process files and send data
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

            # If there's enough data left in the file to read 30 rows
            if start_row < len(df):
                has_data = True
                # Read a fixed 30 rows of data
                data_chunk = df.iloc[start_row:end_row]
                variable_batch_size = 30 if random.random() < 0.75 else random.randint(20, 30)
                variable_batch_size = min(variable_batch_size,
                                          len(data_chunk))  # Ensure it doesn't exceed available rows

                selected_indices = sorted(random.sample(range(len(data_chunk)), variable_batch_size))

                # Select the rows based on the indices
                data_to_send = data_chunk.iloc[selected_indices].to_dict(orient="records")

                # List of columns to randomly remove
                columns_to_remove = [
                    "Gyroscope X (deg/s)", "Gyroscope Y (deg/s)", "Gyroscope Z (deg/s)",
                    "Accelerometer X (g)", "Accelerometer Y (g)", "Accelerometer Z (g)"
                ]

                # Randomly remove values for each column independently
                for column in columns_to_remove:
                    rows_to_remove = random.sample(range(len(data_to_send)),
                                                   random.randint(1, min(3, len(data_to_send))))
                    for i in rows_to_remove:
                        if column in data_to_send[i]:
                            del data_to_send[i][column]

                # Publish the batch to MQTT
                payload = {
                    "file_name": os.path.basename(file_path),
                    "batch_start": start_row + 1,
                    "batch_end": start_row + variable_batch_size,
                    "data": data_to_send
                }
                client.publish(mqtt_topic, json.dumps(payload))
                print(
                    f"Published rows {start_row + 1}-{start_row + variable_batch_size} from {os.path.basename(file_path)}")

                # Update the row counter for the file
                row_counters[idx] += batch_size

        if not has_data:
            print("All files processed.")
            break  # Exit the loop if no data is left to process in any file

# Main execution
if __name__ == "__main__":
    client.loop_start()
    try:
        process_files(folder_path, batch_size)
    finally:
        client.loop_stop()
        client.disconnect()
