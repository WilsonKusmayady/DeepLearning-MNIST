import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist # Menggunakan fashion_mnist
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Sesuaikan nama file model kembali ke Fashion MNIST
MODEL_FILENAME = "model_fashion_mnist.keras" 

# === 1. CEK DAN MUAT MODEL ===
if not os.path.exists(MODEL_FILENAME):
    print(f"Error: File model '{MODEL_FILENAME}' tidak ditemukan.")
    print("Silakan jalankan skrip 'train_model.py' untuk Fashion MNIST terlebih dahulu.")
    sys.exit()

print(f"Memuat model dari '{MODEL_FILENAME}'...")
loaded_model = tf.keras.models.load_model(MODEL_FILENAME)
print("Model berhasil dimuat.")

# === 2. MEMUAT DATA UJI FASHION MNIST DARI KERAS API ===
print("Memuat data uji Fashion MNIST...")
# Kita hanya perlu data test, jadi data train kita abaikan dengan '_'
(_, _), (test_images, test_labels) = fashion_mnist.load_data()

# Pra-pemrosesan data uji
# Menambahkan dimensi channel dan normalisasi nilai piksel
test_images_processed = np.expand_dims(test_images, -1).astype('float32') / 255.0

print(f"Data uji berhasil dimuat. Jumlah gambar: {len(test_images_processed)}")

# Definisikan nama kelas kembali ke nama pakaian
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# === 3. LAKUKAN PREDIKSI ===
print("\nMelakukan prediksi pada data uji...")
predictions = loaded_model.predict(test_images_processed)
print("Prediksi selesai.")


# === 4. VISUALISASI HASIL ===
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False); plt.xticks([]); plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    confidence = 100 * np.max(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    
    # Label plot menggunakan class_names pakaian
    plt.xlabel(f"{class_names[predicted_label]} ({confidence:.0f}%) | Asli: {class_names[true_label]}", color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False); plt.xticks(range(10)); plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    # Gunakan 'test_images' (data asli) untuk plotting
    plot_image(i, predictions[i], test_labels, test_images) 
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show() 