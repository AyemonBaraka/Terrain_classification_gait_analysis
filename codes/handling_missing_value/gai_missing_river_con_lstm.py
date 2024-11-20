from kafka import KafkaConsumer
import json
from river import metrics, preprocessing, compose
import matplotlib.pyplot as plt
import numpy as np
from deep_river.classification import Classifier
from torch import nn
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import time

# Kafka Configuration
bootstrap_servers = "localhost:9092"  # Kafka broker address
topic_name = "features-topic-missing"  # Kafka topic for features

# Initialize River Model and Metrics
accuracy = metrics.Accuracy()  # Accuracy metric
f1_score = metrics.MacroF1()  # Macro F1 score for multi-class classification
roc_auc = metrics.ROCAUC()  # ROC AUC metric

# Preprocessing for feature scaling
scaler = preprocessing.StandardScaler()

# Define the LSTM-based Model
class MyLSTMModel(nn.Module):
    def __init__(self, n_features, hidden_size=128, n_classes=3):
        super(MyLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=-1)  # Softmax for multi-class probabilities

    def forward(self, x):
        # LSTM expects input of shape (batch_size, seq_len, input_size)
        x = x.unsqueeze(0)  # Add batch dimension
        _, (hidden, _) = self.lstm(x)  # Only take the last hidden state
        output = self.fc(hidden[-1])  # Fully connected layer
        return self.softmax(output)


model = compose.Pipeline(
    preprocessing.StandardScaler(),
    Classifier(
        module=MyLSTMModel,
        loss_fn="cross_entropy",
        optimizer_fn="adam",
        lr=0.0005,
        output_is_logit=False
    )
)


# Initialize confusion matrix for sensitivity and specificity
confusion_matrix = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})

# Metrics storage for plotting
macro_roc_aucs, micro_roc_aucs, accuracies, f1_scores = [], [], [], []
sensitivities_per_class = defaultdict(list)  # Store sensitivities for each class
specificities_per_class = defaultdict(list)

# Kafka Consumer Configuration
consumer = KafkaConsumer(
    topic_name,
    bootstrap_servers=bootstrap_servers,
    auto_offset_reset="earliest",
    group_id="ml-consumer-group",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("Kafka consumer is running and listening for features...")

# Global variables for message tracking
y_true_all = []
y_pred_all = []
y_pred_proba_all = []
count = 0
last_message_time = time.time()
timeout = 45  # 45-second timeout
def one_hot_encode(label, num_classes):
    """One-hot encodes a label for the given number of classes."""
    one_hot = [0] * num_classes
    one_hot[label] = 1
    return one_hot

def process_message(features):
    """Processes a single Kafka message."""
    global count, y_true_all, y_pred_all, y_pred_proba_all

    # Extract the target
    target = features.pop("Surface_type", None)

    if target is None:
        print("No target label found in message. Skipping...")
        return

    # Convert target to int
    y_true = int(target)

    # Convert all feature values to floats
    features = {k: float(v) for k, v in features.items()}

    # Predict the target
    y_pred_proba = model.predict_proba_one(features)
    y_pred = max(y_pred_proba, key=y_pred_proba.get)
    n_cls = 3  # Assume 3 classes for this example

    # Update model with the current example
    model.learn_one(features, y_true)

    # Append true labels and predictions for ROC AUC
    if isinstance(y_pred, (int, float)):
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        y_pred_proba_all.append([y_pred_proba.get(cls, 0) for cls in range(n_cls)])
        accuracy.update(y_true, y_pred)
        f1_score.update(y_true, y_pred)

        # Update confusion matrix and calculate sensitivity for each class
        for cls in range(n_cls):
            if y_true == cls and y_pred == cls:
                confusion_matrix[cls]["TP"] += 1
            elif y_true == cls and y_pred != cls:
                confusion_matrix[cls]["FN"] += 1
            elif y_true != cls and y_pred == cls:
                confusion_matrix[cls]["FP"] += 1
            elif y_true != cls and y_pred != cls:
                confusion_matrix[cls]["TN"] += 1

            # Sensitivity calculation
            tp = confusion_matrix[cls]["TP"]
            fn = confusion_matrix[cls]["FN"]
            tn = confusion_matrix[cls]["TN"]
            fp = confusion_matrix[cls]["FP"]

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            sensitivities_per_class[cls].append(sensitivity * 100)

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities_per_class[cls].append(specificity * 100)


        if len(y_true_all) > 100:
            y_true_np = np.array(y_true_all)
            y_pred_proba_np = np.array(y_pred_proba_all)
            roc_auc_per_class = []
            for class_idx in range(n_cls):
                binary_y_true = (y_true_np == class_idx).astype(int)
                class_probabilities = y_pred_proba_np[:, class_idx]
                auc_score = roc_auc_score(binary_y_true, class_probabilities)
                roc_auc_per_class.append(auc_score)

            macro_roc_auc = np.mean(roc_auc_per_class)
            macro_roc_aucs.append(macro_roc_auc * 100)

        accuracies.append(accuracy.get() * 100)
        f1_scores.append(f1_score.get() * 100)

    count += 1
    if count % 1000 == 0 or count > 12750:
        print(f"Processed {count} messages")
        # Print sensitivities for each class
        for cls in range(n_cls):
            cls_dict = {0: "FLAT",
                        1: "GRASS",
                        2: "GRAVEL"}
            print(f"Class {cls_dict[cls]}:" )
            sensitivity_last = sensitivities_per_class[cls][-1] if sensitivities_per_class[cls] else 0
            print(f"            Sensitivity: {sensitivity_last:.2f}%")
            specificities_last = specificities_per_class[cls][-1] if specificities_per_class[cls] else 0
            print(f"            Specificities: {specificities_last:.2f}%")

        print(f"Accuracy: {accuracy.get() * 100:.2f}%")
        print(f"F1 Score: {f1_score.get() * 100:.2f}%")
        print(f"Macro ROC AUC: {macro_roc_auc * 100:.2f}%")


def plot_metrics():
    """Plots the metrics."""
    plt.figure(figsize=(10, 6))

    # Plot sensitivities for each class
    for cls, sens_values in sensitivities_per_class.items():
        plt.plot(sens_values, label=f"Sensitivity (Class {cls})")

    for cls, spes_values in specificities_per_class.items():
        plt.plot(spes_values, label=f"Specificities (Class {cls})")

    plt.plot(accuracies, label="Accuracy")
    plt.plot(f1_scores, label="F1 Score")
    plt.plot(macro_roc_aucs, label="Macro ROC AUC")
    plt.xlabel("Batches of 1000 Messages")
    plt.ylabel("Percentage")
    plt.title("Model Performance Metrics Over Time")
    plt.legend()
    plt.show()

# Kafka Consumer Execution
try:
    print("Starting Kafka Consumer...")
    while True:
        # Poll for new messages
        msg_pack = consumer.poll(timeout_ms=1000)

        if msg_pack:
            for tp, messages in msg_pack.items():
                for message in messages:
                    last_message_time = time.time()
                    process_message(message.value)

        # Check for timeout
        if time.time() - last_message_time >= timeout:
            print("No messages received for 45 seconds. Shutting down and plotting metrics.")
            plot_metrics()
            break

except KeyboardInterrupt:
    print("Consumer interrupted manually. Exiting...")

except Exception as e:
    print(f"Error processing message: {e}")

finally:
    consumer.close()
