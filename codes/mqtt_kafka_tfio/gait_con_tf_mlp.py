import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt

# Kafka Configuration
KAFKA_TOPIC = "gait-test"
KAFKA_SERVERS = "localhost:9092"
GROUP_ID = "cgonline"
NUM_COLUMNS = 10  # Adjust this based on your dataset's number of features

# Model Configuration
BATCH_SIZE = 32
EPOCHS = 5  # Number of epochs for each mini-dataset
NUM_CLASSES = 3  # Adjust based on the number of output classes
SHUFFLE_BUFFER = 32

# Label Mapping
LABEL_MAPPING = {0: "Flat", 1: "Grass", 2: "Gravel"}

# Metrics Storage
val_accuracies = []
val_losses = []

# Counter for processed mini-datasets
mini_ds_count = 0

# Kafka Dataset
online_train_ds = tfio.experimental.streaming.KafkaBatchIODataset(
    topics=[KAFKA_TOPIC],
    group_id=GROUP_ID,
    servers=KAFKA_SERVERS,
    stream_timeout=1000000,
    configuration=[
        "session.timeout.ms=7000",
        "max.poll.interval.ms=8000",
        "auto.offset.reset=earliest"
    ],
)

# Function to decode Kafka messages
def decode_kafka_online_item(raw_message, raw_key):
    # Decode the message as CSV
    message = tf.io.decode_csv(raw_message, [[0.0] for _ in range(NUM_COLUMNS)])
    key = tf.strings.to_number(raw_key, out_type=tf.int32)  # Convert key to integer
    key_one_hot = tf.one_hot(key, depth=NUM_CLASSES)  # One-hot encode assuming NUM_CLASSES classes
    return (tf.stack(message), key_one_hot)

# Define the Model
def build_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize the model
model = build_model(NUM_COLUMNS, NUM_CLASSES)

# Continuous processing loop
print("Starting Kafka Online Learning...")
while True:
    for mini_ds in online_train_ds:
        # Shuffle and batch the mini-dataset
        mini_ds = mini_ds.shuffle(buffer_size=SHUFFLE_BUFFER)
        mini_ds = mini_ds.map(decode_kafka_online_item)
        mini_ds = mini_ds.batch(BATCH_SIZE)

        # Increment the mini-dataset counter
        mini_ds_count += 1

        # Evaluate the model on the mini-dataset
        val_loss, val_accuracy = model.evaluate(mini_ds, verbose=1)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Mini-Dataset {mini_ds_count}: Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")

        # Train the model on the mini-dataset
        model.fit(mini_ds, epochs=EPOCHS, verbose=1)

        # Analyze predictions for each batch
        for val_batch_data, val_batch_labels in mini_ds:
            predictions = model.predict(val_batch_data)
            predicted_labels = np.argmax(predictions, axis=1)  # Predicted class labels
            actual_labels = np.argmax(val_batch_labels.numpy(), axis=1)  # Actual labels

            # Print actual vs. predicted labels
            for i in range(len(predicted_labels)):
                print(f"Actual: {LABEL_MAPPING[actual_labels[i]]}, Predicted: {LABEL_MAPPING[predicted_labels[i]]}")

        # Plot metrics after processing every 5 mini-datasets
        if mini_ds_count % 5 == 0:
            plt.plot(range(1, mini_ds_count + 1), val_accuracies, marker='o', label="Validation Accuracy")
            plt.xlabel("Mini-Dataset Number")
            plt.ylabel("Accuracy")
            plt.title("Validation Accuracy Over Mini-Datasets")
            plt.legend()
            plt.show()
