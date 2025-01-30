import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Fungsi untuk membaca citra grayscale
def read_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Fungsi untuk menghitung fitur tekstur menggunakan GLCM
def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features = {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'correlation': graycoprops(glcm, 'correlation')[0, 0]
    }
    return features

# Path ke citra
image_paths = ['ss.jpg', 'dk.jpg', 'dh.jpg']

# Ekstrak fitur dari setiap citra
for i, path in enumerate(image_paths):
    image = read_image(path)
    features = extract_glcm_features(image)
    print(f'Fitur GLCM untuk Image {i+1}:')
    for key, value in features.items():
        print(f'{key.capitalize()}: {value:.4f}')
    print('-' * 30)
