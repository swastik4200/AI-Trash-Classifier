import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Dataset path
data_dir = "trashnet/data/dataset-resized/dataset-resized"

# Filter valid classes (ignore __MACOSX etc.)
valid_classes = [d for d in os.listdir(data_dir)
                 if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]

print("✅ Using classes:", valid_classes)

# Image size & batch size
img_height, img_width = 150, 150
batch_size = 32

# Data augmentation & split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    classes=valid_classes,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    classes=valid_classes,
    class_mode='categorical',
    subset='validation'
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(valid_classes), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# Save model
model.save("trash_classifier.h5")
print("🎉 Model saved as trash_classifier.h5")
