import json
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import paho.mqtt.client as mqtt
from kafka import KafkaProducer, KafkaConsumer
from scipy.stats import kurtosis, skew
import logging
# MQTT Configuration
mqtt_broker = "10.87.44.24"  # MQTT broker address
mqtt_port = 1883  # MQTT port
mqtt_topic = "gait-missing"  # MQTT topic
mqtt_username = "ayemon12345"  # MQTT username
mqtt_password = "abc12345"  # MQTT password

# Kafka Configuration
kafka_broker = "localhost:9092"  # Kafka broker address
kafka_topic = "raw-data-topic-missing"  # Kafka topic for raw data
features_topic = "features-topic-missing"  # Kafka topic for extracted features

# Surface Type Encoding
surface_type_encoding = {"FLAT": 0, "GRASS": 1, "GRAVEL": 2}

# Kafka Producer for Features
producer = KafkaProducer(
    bootstrap_servers=kafka_broker,
    value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8')
)

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker!")
        client.subscribe(mqtt_topic)
    else:
        print(f"Failed to connect to MQTT broker, return code {rc}")


def on_message(client, userdata, msg):
    try:
        # Decode MQTT message
        payload = json.loads(msg.payload.decode())
        print(payload)

        # Forward data to Kafka
        producer.send(kafka_topic, value=payload)
    except Exception as e:
        print(f"Error processing MQTT message: {e}")


# MQTT Client Setup
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(mqtt_username, mqtt_password)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message


# Feature Extraction Function
def compute_features(data, surface_type):
    features = {}

    # Map Surface Type
    features["Surface_type"] = surface_type_encoding.get(surface_type, -1)  # Default to -1 if unknown

    for axis in ['GX', 'GY', 'GZ', 'AX', 'AY', 'AZ']:
        series = data[axis]

        # Basic statistics
        features[f"Mean{axis}"] = np.mean(series)
        features[f"Var{axis}"] = np.var(series)
        features[f"Std{axis}"] = np.std(series)
        features[f"RMS{axis}"] = np.sqrt(np.mean(series ** 2))
        features[f"Kurtosis{axis}"] = kurtosis(series)
        features[f"Skewness{axis}"] = skew(series)

        # Peak-to-Peak
        features[f"PeakToPeak{axis}"] = np.ptp(series)

        # Frequency Domain Features
        fft = np.fft.fft(series)
        power_spectrum = np.abs(fft) ** 2
        dominant_freq = np.argmax(power_spectrum[:len(power_spectrum) // 2])
        power_dominant_freq = power_spectrum[dominant_freq]
        features[f"DominantFreq{axis}"] = dominant_freq
        features[f"PowerDominantFreq{axis}"] = power_dominant_freq

    # Combined Features
    features["SMAA"] = np.mean(data[["AX", "AY", "AZ"]].abs().sum(axis=1))
    return features


# Kafka Consumer
consumer = KafkaConsumer(
    kafka_topic,
    bootstrap_servers=kafka_broker,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))  # Deserialize JSON messages
)


# Kafka Feature Processing Loop
def process_features():
    print("Kafka consumer is running and listening for messages...")
    count = 0

    last_valid_values = {"GX": None, "GY": None, "GZ": None, "AX": None, "AY": None, "AZ": None}

    for message in consumer:
        # Parse the received batch
        raw_data = message.value
        batch_data = raw_data.get("data", [])

        # Discard batches with fewer than 20 rows
        if len(batch_data) < 22:
            print("Batch discarded (fewer than 22 rows).")
            continue

        # Convert batch data to DataFrame
        df = pd.DataFrame(batch_data)

        # Extract Surface Type
        surface_type = df["Surface_type"].iloc[0] if "Surface_type" in df else "Unknown"

        # Rename columns for feature calculation
        df.rename(columns={
            "Gyroscope X (deg/s)": "GX",
            "Gyroscope Y (deg/s)": "GY",
            "Gyroscope Z (deg/s)": "GZ",
            "Accelerometer X (g)": "AX",
            "Accelerometer Y (g)": "AY",
            "Accelerometer Z (g)": "AZ"
        }, inplace=True)

        for axis in ["GX", "GY", "GZ", "AX", "AY", "AZ"]:
            if axis in df.columns:
                df[axis] = df[axis].fillna(method="ffill")
                if last_valid_values[axis] is not None:
                    df[axis] = df[axis].fillna(last_valid_values[axis])
                last_valid_values[axis] = df[axis].iloc[-1]
            else:
                logger.warning(f"Column {axis} missing, setting to default.")
                df[axis] = 0


        # Compute features for the batch
        features = compute_features(df, surface_type)
        count += 1

        # Print the features
        #print("\nExtracted Features:")
        for key, value in features.items():
            #print(f"{key}: {value}")
            pass

        # Forward features to Kafka
        producer.send(features_topic, value=features)
        print(f"{count} no message: Forwarded features to Kafka topic '{features_topic}'")


# Run MQTT-to-Kafka Bridge and Feature Processing
if __name__ == "__main__":
    try:
        # Start MQTT client in a separate thread
        mqtt_client.connect(mqtt_broker, mqtt_port, 60)
        mqtt_client.loop_start()

        # Process features from Kafka
        process_features()
    except KeyboardInterrupt:
        print("Stopping...")
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
