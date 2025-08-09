import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import os

# ✅ Try to use GPU, but fallback to CPU if not available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("✅ GPU is available and will be used:", physical_devices[0])
    except:
        print("⚠️ GPU setup failed. Proceeding with CPU.")
else:
    print("⚠️ GPU not found. Training will use CPU.")

# ✅ Paths
train_path = "split/train"
val_path = "split/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ✅ Data Generators
train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# ✅ Model Architecture
base = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation='softmax')
])

# ✅ Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train
model.fit(train_data, validation_data=val_data, epochs=10)

# ✅ Save Model and Labels
model.save('dog_breed_model.h5')
np.save("class_labels.npy", np.array(list(train_data.class_indices.keys())))

print("✅ Training complete. Model and class labels saved.")
