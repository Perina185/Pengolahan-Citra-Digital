import numpy as np
from skimage.feature import graycomatrix

# Fungsi manual untuk menghitung properti GLCM
def calculate_glcm_properties(glcm):
    properties = {}
    
    # Kontras
    i, j = np.ogrid[0:glcm.shape[0], 0:glcm.shape[1]]
    contrast = np.sum(glcm * (i - j) ** 2)
    properties['contrast'] = contrast
    
    # Energi
    energy = np.sum(glcm ** 2)
    properties['energy'] = energy
    
    # Homogenitas
    homogenity = np.sum(glcm / (1.0 + np.abs(i - j)))
    properties['homogeneity'] = homogenity
    
    # Korelasi
    i_mean = np.sum(i * np.sum(glcm, axis=1))
    j_mean = np.sum(j * np.sum(glcm, axis=0))
    i_std = np.sqrt(np.sum((i - i_mean) ** 2 * np.sum(glcm, axis=1)))
    j_std = np.sqrt(np.sum((j - j_mean) ** 2 * np.sum(glcm, axis=0)))
    correlation = np.sum((i - i_mean) * (j - j_mean) * glcm) / (i_std * j_std + 1e-10)
    properties['correlation'] = correlation
    
    return properties

# Contoh gambar (matriks intensitas)
image = np.array([[0, 1, 2],
                  [2, 2, 0],
                  [1, 0, 0]])

# Buat GLCM
glcm = graycomatrix(image, distances=[1], angles=[0], levels=3, symmetric=True, normed=True)

# Karena GLCM mengembalikan array 4D, ambil level pertama [0, 0]
glcm_matrix = glcm[:, :, 0, 0]

# Hitung properti GLCM
glcm_properties = calculate_glcm_properties(glcm_matrix)

# Tampilkan hasil
for prop, value in glcm_properties.items():
    print(f"{prop.capitalize()}: {value:.4f}")