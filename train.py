import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist # Menggunakan fashion_mnist
from tensorflow.keras import layers, models
import numpy as np
import os

# Nama file model dikembalikan ke Fashion MNIST
MODEL_FILENAME = "model_fashion_mnist.keras"

# === 1. MEMUAT DATA TRAINING FASHION MNIST DARI KERAS API ===
print("Memuat data training Fashion MNIST...")
# Kita hanya perlu data training, jadi data test kita abaikan dengan '_'
(train_images, train_labels), (_, _) = fashion_mnist.load_data()

# Pra-pemrosesan data
# Menambahkan dimensi channel dan normalisasi
train_images = np.expand_dims(train_images, -1).astype('float32') / 255.0

print(f"Data training berhasil dimuat. Jumlah gambar: {len(train_images)}")
print(f"Bentuk data setelah diproses: {train_images.shape}")


# === 2. MEMBUAT ATAU MEMUAT MODEL ===
if os.path.exists(MODEL_FILENAME):
    print(f"File model '{MODEL_FILENAME}' ditemukan. Melanjutkan training...")
    model = tf.keras.models.load_model(MODEL_FILENAME)
else:
    print(f"File model '{MODEL_FILENAME}' tidak ditemukan. Membuat model baru.")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(10, activation='softmax')
    ])

model.summary()

# === 3. COMPILE MODEL ===
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === 4. MELATIH MODEL ===
print("\nMemulai proses training...")
model.fit(train_images, train_labels, 
          epochs=20,  
          batch_size=32,
          validation_split=0.1)
print("Training selesai!")

# === 5. SIMPAN MODEL ===
print(f"\nMenyimpan model ke '{MODEL_FILENAME}'...")
model.save(MODEL_FILENAME)
print("Model berhasil disimpan.")