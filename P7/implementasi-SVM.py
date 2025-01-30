import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Membaca dataset citra
# Pastikan Anda memiliki array X (fitur citra) dan y (label kelas)
# Contoh dummy data:
X = np.random.rand(100, 64)  # 100 sampel dengan 64 fitur
y = np.random.randint(0, 2, 100)  # 100 label dengan 2 kelas (0 atau 1)

# Split data untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inisialisasi model SVM
clf = svm.SVC(kernel='linear', random_state=42)

# Latih model menggunakan data pelatihan
clf.fit(X_train, y_train)

# Prediksi menggunakan data uji
y_pred = clf.predict(X_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:.2f}%')
