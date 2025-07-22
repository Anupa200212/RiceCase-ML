# RiceCase-ML
# file braking code
import os
import shutil
import random

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set source and destination directories
original_dir = '/content/drive/MyDrive/testRice/original_data'
base_dir = '/content/drive/MyDrive/testRice/split_data'

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

classes = ['Brown Spot', 'Others']  # Add more if needed

# Function to create folders
def create_dirs():
    for split in [train_dir, val_dir, test_dir]:
        for cls in classes:
            os.makedirs(os.path.join(split, cls), exist_ok=True)

# Function to split and move files
def split_data():
    for cls in classes:
        src_folder = os.path.join(original_dir, cls)
        all_files = os.listdir(src_folder)
        random.shuffle(all_files)

        total = len(all_files)
        train_count = int(0.7 * total)
        val_count = int(0.15 * total)

        train_files = all_files[:train_count]
        val_files = all_files[train_count:train_count + val_count]
        test_files = all_files[train_count + val_count:]

        for file_set, target_dir in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:
            for file in file_set:
                src = os.path.join(src_folder, file)
                dst = os.path.join(target_dir, cls, file)
                shutil.copy2(src, dst)

# Run functions
create_dirs()
split_data()

print("✅ Dataset split completed.")














# 📦 Import necessary libraries
from google.colab import drive
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 🔹 Mount Google Drive
drive.mount('/content/drive')

# 🔹 Define dataset paths
base_path = "/content/drive/MyDrive/testRice/split_data"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")
test_path = os.path.join(base_path, "test")

# 🔹 Define paths to save model and results
model_save_path = "/content/drive/MyDrive/testRice/result/cat_dog_CNN_model.h5"
results_save_path = "/content/drive/MyDrive/testRice/result/cat_dog_CNN_results.txt"
conf_matrix_path = "/content/drive/MyDrive/testRice/result/cat_dog_conf_matrix.png"

# 🔹 Image size and batch size
img_size = 224
batch_size = 32

# 🔹 Data Generators
# Augment only training data
train_datagen = ImageDataGenerator(
    rescale=1./255,               # Normalize pixel values
    rotation_range=20,            # Randomly rotate images
    width_shift_range=0.2,        # Randomly shift images horizontally
    height_shift_range=0.2,       # Randomly shift images vertically
    shear_range=0.2,              # Random shear transformation
    zoom_range=0.2,               # Random zoom
    horizontal_flip=True,         # Random horizontal flip
    fill_mode='nearest'           # Fill in new pixels
)

# No augmentation for validation and test sets
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 🔹 Load data from directories
train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False  # Important for evaluation
)

# 🔹 Define CNN model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),  # 1st conv layer
    MaxPooling2D(2,2),                                                         # 1st pooling layer
    Conv2D(64, (3,3), activation='relu'),                                      # 2nd conv layer
    MaxPooling2D(2,2),                                                         # 2nd pooling layer
    Conv2D(128, (3,3), activation='relu'),                                     # 3rd conv layer
    MaxPooling2D(2,2),                                                         # 3rd pooling layer
    Flatten(),                                                                 # Flatten feature maps
    Dense(256, activation='relu'),                                             # Fully connected layer
    Dropout(0.5),                                                              # Dropout for regularization
    Dense(train_data.num_classes, activation='softmax')                        # Output layer
])

# 🔹 Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 🔹 Show model summary
model.summary()

# 🔹 Train the model
epochs = 10
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# 🔹 Save the trained model
model.save(model_save_path)
print(f"✅ Model saved at: {model_save_path}")

# 🔹 Evaluate on train/val/test datasets
train_loss, train_acc = model.evaluate(train_data)
val_loss, val_acc = model.evaluate(val_data)
test_loss, test_acc = model.evaluate(test_data)

# 🔹 Print accuracies
print(f"🔹 Training Accuracy: {train_acc*100:.2f}%")
print(f"🔹 Validation Accuracy: {val_acc*100:.2f}%")
print(f"🔹 Test Accuracy: {test_acc*100:.2f}%")

# 🔹 Make predictions on test data
true_labels = test_data.classes
class_labels = list(test_data.class_indices.keys())
pred_probs = model.predict(test_data)
pred_labels = np.argmax(pred_probs, axis=1)

# 🔹 Classification report and confusion matrix
report = classification_report(true_labels, pred_labels, target_names=class_labels)
cm = confusion_matrix(true_labels, pred_labels)

# 🔹 Save results to a file
with open(results_save_path, "w") as f:
    f.write(f"Training Accuracy: {train_acc*100:.2f}%\n")
    f.write(f"Validation Accuracy: {val_acc*100:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")
print(f"✅ Results saved at: {results_save_path}")

# 🔹 Plot training/validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.show()

# 🔹 Plot training/validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# 🔹 Plot confusion matrix as heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(conf_matrix_path)
plt.show()
print(f"✅ Confusion matrix saved at: {conf_matrix_path}")





VG16 CNN CODE
-------------
# 📦 Import necessary libraries
from google.colab import drive
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 🔹 Mount Google Drive
drive.mount('/content/drive')

# 🔹 Define dataset paths
base_path = "/content/drive/MyDrive/testRice/split_data"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")
test_path = os.path.join(base_path, "test")

# 🔹 Define paths to save model and results
model_save_path = "/content/drive/MyDrive/testRice/result/cat_dog_VGG16_model.h5"
results_save_path = "/content/drive/MyDrive/testRice/result/cat_dog_VGG16_results.txt"
conf_matrix_path = "/content/drive/MyDrive/testRice/result/cat_dog_VGG16_conf_matrix.png"

# 🔹 Image size and batch size
img_size = 224
batch_size = 32

# 🔹 Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path, target_size=(img_size, img_size),
    batch_size=batch_size, class_mode="categorical"
)
val_data = val_datagen.flow_from_directory(
    val_path, target_size=(img_size, img_size),
    batch_size=batch_size, class_mode="categorical"
)
test_data = test_datagen.flow_from_directory(
    test_path, target_size=(img_size, img_size),
    batch_size=batch_size, class_mode="categorical", shuffle=False
)

# 🔹 Load VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze convolutional layers

# 🔹 Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

# 🔹 Final model
model = Model(inputs=base_model.input, outputs=output)

# 🔹 Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 🔹 Train the model
epochs = 100
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# 🔹 Save the model
model.save(model_save_path)
print(f"✅ Model saved at: {model_save_path}")

# 🔹 Evaluate
train_loss, train_acc = model.evaluate(train_data)
val_loss, val_acc = model.evaluate(val_data)
test_loss, test_acc = model.evaluate(test_data)

print(f"🔹 Training Accuracy: {train_acc*100:.2f}%")
print(f"🔹 Validation Accuracy: {val_acc*100:.2f}%")
print(f"🔹 Test Accuracy: {test_acc*100:.2f}%")

# 🔹 Predictions and Evaluation
true_labels = test_data.classes
class_labels = list(test_data.class_indices.keys())
pred_probs = model.predict(test_data)
pred_labels = np.argmax(pred_probs, axis=1)

report = classification_report(true_labels, pred_labels, target_names=class_labels)
cm = confusion_matrix(true_labels, pred_labels)

# 🔹 Save results to a file
with open(results_save_path, "w") as f:
    f.write(f"Training Accuracy: {train_acc*100:.2f}%\n")
    f.write(f"Validation Accuracy: {val_acc*100:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")
print(f"✅ Results saved at: {results_save_path}")

# 🔹 Accuracy Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.show()

# 🔹 Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# 🔹 Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(conf_matrix_path)
plt.show()
print(f"✅ Confusion matrix saved at: {conf_matrix_path}")
