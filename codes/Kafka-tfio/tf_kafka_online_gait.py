import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt

# Define the number of columns (features) in your dataset
NUM_COLUMNS = len([
    'MeanGX', 'MeanGY', 'MeanGZ',
    'VarGX', 'VarGY', 'VarGZ',
    'StdGX', 'StdGY', 'StdGZ',
    'RMSGX', 'RMSGY', 'RMSGZ',
    'KurtosisGX', 'KurtosisGY', 'KurtosisGZ',
    'SkewnessGX', 'SkewnessGY', 'SkewnessGZ',
    'ZCRGX', 'ZCRGY', 'ZCRGZ',
    'PeakToPeakGX', 'PeakToPeakGY', 'PeakToPeakGZ',
    'DominantFreqGX', 'DominantFreqGY', 'DominantFreqGZ',
    'PowerDominantFreqGX', 'PowerDominantFreqGY', 'PowerDominantFreqGZ',
    'EntropyGX', 'EntropyGY', 'EntropyGZ',
    'SpectralEntropyGX', 'SpectralEntropyGY', 'SpectralEntropyGZ',
    'PSDGX', 'PSDGY', 'PSDGZ',
    'MeanAX', 'MeanAY', 'MeanAZ',
    'VarAX', 'VarAY', 'VarAZ',
    'StdAX', 'StdAY', 'StdAZ',
    'RMSAX', 'RMSAY', 'RMSAZ',
    'KurtosisAX', 'KurtosisAY', 'KurtosisAZ',
    'SkewnessAX', 'SkewnessAY', 'SkewnessAZ',
    'ZCRAX', 'ZCRAY', 'ZCRAZ',
    'PeakToPeakAX', 'PeakToPeakAY', 'PeakToPeakAZ',
    'DominantFreqAX', 'DominantFreqAY', 'DominantFreqAZ',
    'PowerDominantFreqAX', 'PowerDominantFreqAY', 'PowerDominantFreqAZ',
    'EntropyAX', 'EntropyAY', 'EntropyAZ',
    'SpectralEntropyAX', 'SpectralEntropyAY', 'SpectralEntropyAZ',
    'PSDAX', 'PSDAY', 'PSDAZ',
    'SMAA', 'TiltA'
])

# Load the saved MLP model
model_path = "/home/ayemon/KafkaProjects/kafkaspark07_90/gait/model/tf_mlp"
model = tf.keras.models.load_model(model_path)

# Kafka streaming configuration
online_train_ds = tfio.experimental.streaming.KafkaBatchIODataset(
    topics=["gait-test"],
    group_id="cgonline",
    servers="localhost:9092",
    stream_timeout=1000000,
    configuration=[
        "session.timeout.ms=7000",
        "max.poll.interval.ms=8000",
        "auto.offset.reset=earliest"
    ],
)

# Function to decode the Kafka messages
def decode_kafka_online_item(raw_message, raw_key):
    # Decode the message as CSV
    message = tf.io.decode_csv(raw_message, [[0.0] for _ in range(NUM_COLUMNS)])
    key = tf.strings.to_number(raw_key, out_type=tf.int32)  # Convert key to integer
    key_one_hot = tf.one_hot(key, depth=3)  # One-hot encode assuming 3 classes
    return (tf.stack(message), key_one_hot)

# Store validation metrics
val_accuracies = []
val_losses = []

# Label mapping
label_mapping = {0: "Flat", 1: "Grass", 2: "Gravel"}

# Counter for mini-datasets
mini_ds_count = 0

# Continuous processing loop
while True:
    # Process Kafka batches and perform online training
    for mini_ds in online_train_ds:
        # Shuffle and batch the mini-dataset
        mini_ds = mini_ds.shuffle(buffer_size=32)
        mini_ds = mini_ds.map(decode_kafka_online_item)
        mini_ds = mini_ds.batch(32)

        if len(mini_ds) > 0:
            # Increment the mini-dataset counter
            mini_ds_count += 1

            # Evaluate the current model on the mini-dataset
            val_loss, val_accuracy = model.evaluate(mini_ds, verbose=1)

            # Store evaluation metrics
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

            # Train the model on the mini-dataset
            model.fit(mini_ds, epochs=25)

            # Analyze predictions for each batch
            for val_batch_data, val_batch_labels in mini_ds:
                predictions = model.predict(val_batch_data)
                predicted_labels = np.argmax(predictions, axis=1)  # Predicted class labels
                actual_labels = np.argmax(val_batch_labels.numpy(), axis=1)  # Actual labels

                # Print actual vs. predicted labels in the desired format
                for i in range(len(predicted_labels)):
                    print(f"Actual: {label_mapping[actual_labels[i]]}, Predicted: {label_mapping[predicted_labels[i]]}")

            # After processing 5 mini-datasets, plot the validation accuracy
            if mini_ds_count % 5 == 0:
                plt.plot(range(1, mini_ds_count + 1), val_accuracies, marker='o', label="Validation Accuracy")
                plt.xlabel("Mini-Dataset Number")
                plt.ylabel("Accuracy")
                plt.title("Validation Accuracy Over Mini-Datasets")
                plt.legend()
                plt.show()
