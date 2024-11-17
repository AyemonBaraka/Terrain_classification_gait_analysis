##working a little bit

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Kafka Configuration
kafka_broker = "localhost:9092"
features_topic = "features-topic"
num_columns = 119  # Number of feature columns
num_classes = 3  # Number of output classes
batch_size = 32

# Define the TensorFlow Model
def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(248, activation='relu', input_shape=(input_dim,)),
        Dense(124, activation='relu'),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize the model
model = build_model(input_dim=num_columns, output_dim=num_classes)

# Kafka streaming configuration
try:
    print("Initializing Kafka Dataset...")
    online_train_ds = tfio.experimental.streaming.KafkaBatchIODataset(
        topics=[features_topic],
        group_id="tensorflow-consumer-group",
        servers=kafka_broker,
        stream_timeout=1000000,
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=earliest"
        ],
    )
    print("Kafka Dataset initialized successfully.")
except Exception as e:
    print(f"Error initializing Kafka Dataset: {e}")
    exit()

# Decode Kafka messages
def decode_kafka_online_item(raw_message, raw_key):
    """Decode Kafka message value and key."""


    # Split the message and count fields
    fields = tf.strings.split(raw_message, ",")
    num_fields = tf.shape(fields)[0]


    # Check if the record is valid
    valid_record = tf.equal(num_fields, num_columns)

    # Use tf.cond for TensorFlow-safe conditional execution
    def decode_valid_record():
        try:
            message = tf.io.decode_csv(raw_message, [[0.0] for _ in range(num_columns)])
            message = tf.stack(message)
            key = tf.strings.to_number(raw_key, out_type=tf.int32)
            key_one_hot = tf.one_hot(key, depth=num_classes)

            return message, key_one_hot
        except Exception as e:
            tf.print("Error decoding Kafka message:", e)
            return tf.zeros([num_columns], dtype=tf.float32), tf.zeros([num_classes], dtype=tf.float32)

    def handle_invalid_record():
        tf.print("Invalid number of fields in record. Skipping...")
        return tf.zeros([num_columns], dtype=tf.float32), tf.zeros([num_classes], dtype=tf.float32)

    return tf.cond(valid_record, decode_valid_record, handle_invalid_record)

# Training loop with validation
val_accuracies = []
val_losses = []
mini_ds_count = 0
max_iterations = 1000
iteration_count = 0

from sklearn.metrics import classification_report, confusion_matrix

print("Starting Kafka TensorFlow Training...")

while iteration_count < max_iterations:
    try:
        for mini_ds in online_train_ds:
            iteration_count += 1

            print(f"Processing mini-dataset {iteration_count}...")

            # Map and filter the dataset
            mini_ds = mini_ds.map(decode_kafka_online_item)
            mini_ds = mini_ds.filter(lambda x, y: not tf.reduce_all(tf.equal(x, 0)) and not tf.reduce_all(tf.equal(y, 0)))
            mini_ds = mini_ds.shuffle(buffer_size=32).batch(batch_size)

            # Check if mini-dataset has data
            dataset_empty = True
            for sample in mini_ds.take(1):
                dataset_empty = False
                break

            if dataset_empty:
                print("Mini-dataset is empty. Skipping...")
                continue

            # Increment mini-dataset counter
            mini_ds_count += 1

            # Evaluate the current model on the mini-dataset
            val_loss, val_accuracy = model.evaluate(mini_ds, verbose=1)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"[Mini-dataset {mini_ds_count}] Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

            # Train the model on the mini-dataset
            model.fit(mini_ds, epochs=25, verbose=1)

            # Get predictions and actual labels for visualization
            actual_labels = []
            predicted_labels = []

            for features, labels in mini_ds:
                preds = model.predict(features, verbose=0)
                actual_labels.extend(tf.argmax(labels, axis=1).numpy())
                predicted_labels.extend(tf.argmax(preds, axis=1).numpy())

            print("Classification Report:")
            print(classification_report(actual_labels, predicted_labels))

            print("Confusion Matrix:")
            print(confusion_matrix(actual_labels, predicted_labels))

            # Plot validation accuracy after every 5 mini-datasets
            if mini_ds_count % 5 == 0:
                plt.plot(range(1, mini_ds_count + 1), val_accuracies, marker='o', label="Validation Accuracy")
                plt.xlabel("Mini-Dataset Number")
                plt.ylabel("Accuracy")
                plt.title("Validation Accuracy Over Mini-Datasets")
                plt.legend()
                plt.show()
    except Exception as e:
        print(f"Error in the training loop: {e}")
        break
