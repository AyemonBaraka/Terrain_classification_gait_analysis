import json
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import paho.mqtt.client as mqtt
from kafka import KafkaProducer, KafkaConsumer
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch
# MQTT Configuration
mqtt_broker = "10.87.44.24"  # MQTT broker address
mqtt_port = 1883  # MQTT port
mqtt_topic = "temp-gait"  # MQTT topic
mqtt_username = "ayemon12345"  # MQTT username
mqtt_password = "abc12345"  # MQTT password

# Kafka Configuration
kafka_broker = "localhost:9092"  # Kafka broker address
kafka_topic = "raw-data-topic"  # Kafka topic for raw data
features_topic = "features-topic"  # Kafka topic for extracted features


count = 0
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

        # Print received MQTT data
        #print("\nReceived data from MQTT:")
        #print(json.dumps(payload, indent=4))

        # Forward data to Kafka
        producer.send(kafka_topic, value=payload)
        #print(f"Forwarded data to Kafka topic '{kafka_topic}'")
    except Exception as e:
        print(f"Error processing MQTT message: {e}")


# MQTT Client Setup
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(mqtt_username, mqtt_password)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message


# Feature Extraction Function
"""def compute_features(data, surface_type):
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

        # Zero Crossing Rate
        features[f"ZCR{axis}"] = ((series[:-1] * series[1:]) < 0).sum()

        # Frequency Domain Features
        fft = np.fft.fft(series)
        power_spectrum = np.abs(fft) ** 2
        dominant_freq = np.argmax(power_spectrum[:len(power_spectrum) // 2])
        power_dominant_freq = power_spectrum[dominant_freq]
        features[f"DominantFreq{axis}"] = dominant_freq
        features[f"PowerDominantFreq{axis}"] = power_dominant_freq

        # Entropy
        prob = power_spectrum / np.sum(power_spectrum)
        entropy = -np.sum(prob * np.log2(prob + 1e-12))
        features[f"Entropy{axis}"] = entropy

    # Combined Features
    features["SMAA"] = np.mean(data[["AX", "AY", "AZ"]].abs().sum(axis=1))
    return features"""



import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch

def compute_features(data, surface_type):
    features = {}

    # Map Surface Type
    surface_type_encoding = {"FLAT": 0, "GRASS": 1, "GRAVEL": 2}  # Define encoding if not globally available
    features["Surface_type"] = surface_type_encoding.get(surface_type, -1)  # Default to -1 if unknown

    for axis in ['GX', 'GY', 'GZ', 'AX', 'AY', 'AZ']:
        series = data[axis]

        # Basic statistics
        features[f"Mean{axis}"] = np.mean(series)
        features[f"Median{axis}"] = np.median(series)
        features[f"Var{axis}"] = np.var(series)
        features[f"Std{axis}"] = np.std(series)
        features[f"RMS{axis}"] = np.sqrt(np.mean(series ** 2))
        features[f"Max{axis}"] = np.max(series)
        features[f"Min{axis}"] = np.min(series)
        features[f"Range{axis}"] = np.ptp(series)
        features[f"Kurtosis{axis}"] = kurtosis(series)
        features[f"Skewness{axis}"] = skew(series)
        features[f"IQR{axis}"] = np.percentile(series, 75) - np.percentile(series, 25)

        # Peak-to-Peak
        features[f"PeakToPeak{axis}"] = np.ptp(series)

        # Zero Crossing Rate
        features[f"ZCR{axis}"] = ((series[:-1] * series[1:]) < 0).sum()

        # Frequency Domain Features
        fft = np.fft.fft(series)
        power_spectrum = np.abs(fft) ** 2
        dominant_freq = np.argmax(power_spectrum[:len(power_spectrum) // 2])
        power_dominant_freq = power_spectrum[dominant_freq]
        features[f"DominantFreq{axis}"] = dominant_freq
        features[f"PowerDominantFreq{axis}"] = power_dominant_freq

        # Entropy
        prob = power_spectrum / np.sum(power_spectrum)
        entropy = -np.sum(prob * np.log2(prob + 1e-12))
        features[f"Entropy{axis}"] = entropy

        # Time-Domain Event-Based Features
        features[f"MeanCrossingRate{axis}"] = ((series[:-1] - np.mean(series)) * (series[1:] - np.mean(series)) < 0).sum()
        series_np = series.to_numpy()  # Convert to NumPy array
        features[f"PeakCount{axis}"] = len(
            np.where((series_np[1:-1] > series_np[:-2]) & (series_np[1:-1] > series_np[2:]))[0]
        )

        # Energy Features
        features[f" {axis}"] = np.sum(series ** 2)

    # Combined Features
    features["SMAA"] = np.mean(data[["AX", "AY", "AZ"]].abs().sum(axis=1))
    features["EuclideanNorm"] = np.sqrt(np.sum(data[["AX", "AY", "AZ"]] ** 2, axis=1)).mean()
    features["Correlation_AX_AY"] = np.corrcoef(data["AX"], data["AY"])[0, 1]
    features["Correlation_AX_AZ"] = np.corrcoef(data["AX"], data["AZ"])[0, 1]
    features["Correlation_AY_AZ"] = np.corrcoef(data["AY"], data["AZ"])[0, 1]

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
    for message in consumer:
        # Parse the received batch
        raw_data = message.value
        batch_data = raw_data.get("data", [])

        if batch_data:
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

            # Compute features for the batch
            features = compute_features(df, surface_type)
            count +=1

            # Print the features
            print("\nExtracted Features:")
            for key, value in features.items():
                print(f"{key}: {value}")

            # Forward features to Kafka
            producer.send(features_topic, value=features)
            print(f" {count} no  message Forwarded features to Kafka topic '{features_topic}'")
        else:
            print("No data in the batch.")


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
