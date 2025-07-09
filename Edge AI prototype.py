# Step 1: Import and prepare data (e.g., recyclable vs. non-recyclable items)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image data setup (sample dataset, e.g., from Kaggle or manually uploaded)
img_size = 64
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    '/content/data/',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')

val_data = train_datagen.flow_from_directory(
    '/content/data/',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')

# Step 2: Build lightweight CNN model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=5)

# Step 3: Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('recyclable_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
