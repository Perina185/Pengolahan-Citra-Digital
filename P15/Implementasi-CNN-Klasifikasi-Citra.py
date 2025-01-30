import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load dataset MNIST sebagai contoh
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisasi data (ubah nilai piksel menjadi rentang 0-1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Menambahkan dimensi channel (karena CNN memerlukan input 4D: (batch_size, height, width, channels))
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Membangun model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Melatih model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluasi model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Akurasi CNN: {test_acc * 100:.2f}%")

# Visualisasi hasil pelatihan
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Validation')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.show()

# Visualisasi beberapa prediksi
predictions = model.predict(x_test)
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_test[i]}\nPred: {predictions[i].argmax()}")
    plt.axis('off')
plt.tight_layout()
plt.show()
