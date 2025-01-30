import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# 1. Load dataset (contoh: digits dataset dari sklearn)
data = load_digits()
X = data.images  # Citra dalam bentuk matriks
y = data.target  # Label kelas

# 2. Flatten citra menjadi vektor
n_samples = len(X)
X = X.reshape((n_samples, -1))  # Ubah dari 2D menjadi 1D

# 3. Normalisasi fitur
X = X / 16.0  # Digit memiliki nilai piksel antara 0-16

# 4. Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Implementasi KNN
k = 5  # Jumlah tetangga terdekat
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 6. Prediksi pada data uji
y_pred = knn.predict(X_test)

# 7. Evaluasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi KNN: {accuracy * 100:.2f}%")

# 8. Visualisasi beberapa prediksi
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Label: {y_test[i]}\nPred: {y_pred[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()