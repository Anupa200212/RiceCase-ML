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

//print("✅ Dataset split completed.")
