import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Correct path to dataset
data_dir = "trashnet/data/dataset-resized/dataset-resized"

# Clean up unwanted folders like __MACOSX
valid_classes = [d for d in os.listdir(data_dir)
                 if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]

print("✅ Using classes:", valid_classes)

# Image preprocessing
img_height, img_width = 150, 150
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    classes=valid_classes,   # ✅ Only use clean classes
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    classes=valid_classes,   # ✅ Only use clean classes
    class_mode='categorical',
    subset='validation'
)
