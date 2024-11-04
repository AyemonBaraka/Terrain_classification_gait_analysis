# Import necessary libraries
from river import stream, compose, preprocessing, metrics
from deep_river.classification import Classifier
import torch
from torch import nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kafka import KafkaConsumer
import time

# Kafka configuration
bootstrap_servers = "localhost:9092"  # Update with your Kafka server address
topic_name = "river-train"  # Kafka topic for the training data

# Initialize the Kafka consumer
consumer = KafkaConsumer(
    topic_name,
    bootstrap_servers=bootstrap_servers,
    auto_offset_reset="earliest",
    group_id="deep-river-consumer-group"
)

# Define a custom PyTorch model
class MyDeepModel(nn.Module):
    def __init__(self, n_features):
        super(MyDeepModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 3),  # Adjust output size for number of classes
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize the deep learning-based classifier
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    Classifier(
        module=MyDeepModel,
        loss_fn="cross_entropy",
        optimizer_fn="adam",
        lr=0.001,
        output_is_logit=False
    )
)

# Metrics
accuracy = metrics.Accuracy()
f1_score = metrics.MacroF1()
f1_scores = []
accuracies = []

# Counter for message tracking
count = 0

# Label mapping
label_mapping = {
    "FLAT": 0,
    "GRASS": 1,
    "GRAVEL": 2
}

# Function to process each Kafka message
def process_message(message_value, message_key):
    global count
    # Convert the message to a feature dictionary
    xi = dict(zip(
        [f"feature_{i}" for i in range(len(message_value.split(",")))],
        map(float, message_value.split(","))
    ))

    # Convert label from string to integer
    yi = label_mapping.get(message_key)
    if yi is None:
        print(f"Unrecognized label: {message_key}")
        return

    # Predict and update metrics
    yi_pred = model.predict_one(xi)
    if yi_pred is not None:
        accuracy.update(yi, yi_pred)
        f1_score.update(yi, yi_pred)
        accuracies.append(accuracy.get() * 100)
        f1_scores.append(f1_score.get() * 100)

    # Increment the counter and print metrics every 1000 messages
    count += 1
    if count % 1000 == 0:
        print(f"Processed {count} messages")
        print(f"Accuracy: {accuracy.get() * 100:.2f}%")
        print(f"F1 Score: {f1_score.get() * 100:.2f}%")

    # Train the model
    model.learn_one(xi, yi)

# Plotting function
def plot(scores, title):
    sns.lineplot(x=range(len(scores)), y=scores)
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.title(title)
    plt.show()

# Consume and process messages from Kafka
last_message_time = time.time()
timeout = 60  # 1 minute timeout

for message in consumer:
    try:
        message_value = message.value.decode("utf-8")
        message_key = message.key.decode("utf-8")
        process_message(message_value, message_key)

        # Update the last message time
        last_message_time = time.time()

        # Check if 1 minute has passed without receiving a message
        if time.time() - last_message_time >= timeout:
            print("No messages received for 1 minute. Plotting metrics...")
            plot(accuracies, "Accuracy Over Time")
            plot(f1_scores, "F1 Score Over Time")
            break  # Exit the loop if timeout occurs

    except Exception as e:
        print(f"Error processing message: {e}")

# Final plot if loop ends
plot(accuracies, "Final Accuracy Over Time")
plot(f1_scores, "Final F1 Score Over Time")

# Print final accuracy
print("Final Accuracy:", accuracy.get())