import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Step 1: Load the gait data
train_file_path = "/home/ayemon/KafkaProjects/Dataset/gait_data/feature_80_20/train_all.csv"
test_file_path = "/home/ayemon/KafkaProjects/Dataset/gait_data/feature_80_20/test_all.csv"

# Load the datasets separately
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Step 2: Encode the target labels (surface_type)
label_encoder = LabelEncoder()
train_df['surface_type'] = label_encoder.fit_transform(train_df['surface_type'])
test_df['surface_type'] = label_encoder.transform(test_df['surface_type'])

# Display the encoding
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Encoding (String to Numeric):", label_mapping)

# Step 3: Split the data into features and target
x_train = train_df.drop(['surface_type'], axis=1).values
y_train = train_df['surface_type'].values

x_test = test_df.drop(['surface_type'], axis=1).values
y_test = test_df['surface_type'].values
# Step 4: Further split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Step 5: Normalize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
# Step 6: One-hot encode the target labels for multi-class classification
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# Step 7: Build the MLP model
model = Sequential()
model.add(Dense(1024, input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(y_train.shape[1], activation='softmax'))  # Output layer

# Step 8: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 9: Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=128,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping],
                    verbose=1)

# Step 10: Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# Step 11: Save the model
model_save_path = "/home/ayemon/KafkaProjects/kafkaspark07_90/gait/model/tf_mlp"
os.makedirs(model_save_path, exist_ok=True)
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Step 12: Make predictions and evaluate
y_pred = np.argmax(model.predict(x_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Step 13: Print a confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
conf_matrix = confusion_matrix(y_test_labels, y_pred)
accuracy = accuracy_score(y_test_labels, y_pred)

print("Test Data Accuracy: {:.4f}".format(accuracy))
print("Confusion Matrix:")
print(conf_matrix)
