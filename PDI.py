# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:55:14 2025

@author: yakit
"""

# =============================================================================
# SCRIPT DE PDI
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.filters import median
from skimage.morphology import disk
from skimage.feature import canny
from scipy import stats

def procesar_mamografia(imagen):
    """
    Procesa una mamografía siguiendo los pasos definidos:
    1. Recorte de bordes.
    2. Conversión a escala de grises.
    3. Eliminación de áreas de interés.
    4. Aplicación de filtro bilateral y de mediana.
    5. Mejora de contraste con CLAHE.
    """
    # Paso 1: Cargar la mamografía original
    imagen = imagen
    # plt.figure()
    # plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    # plt.title('Imagen Original')
    # plt.show()

    # Paso 1.1: Recorte de bordes
    filas, columnas, _ = imagen.shape
    recorte_filas = int(filas * 0.02)
    recorte_columnas = int(columnas * 0.02)
    imagen = imagen[recorte_filas:-recorte_filas, recorte_columnas:-recorte_columnas, :]
    # plt.figure()
    # plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    # plt.title('Recorte de Bordes (2%)')
    # plt.show()

    # Paso 1.2: Detección y recorte de bordes
    perfil_intensidad = np.mean(imagen, axis=(0, 2))  # Promedio en filas y canales
    mitad_izquierda = perfil_intensidad[:len(perfil_intensidad) // 2]
    mitad_derecha = perfil_intensidad[len(perfil_intensidad) // 2:]

    if np.mean(mitad_derecha) > np.mean(mitad_izquierda):
        # Seno orientado hacia la derecha
        recorte_horizontal = np.argmax(perfil_intensidad)
        imagen = np.fliplr(imagen[:, :recorte_horizontal, :])
    else:
        # Seno orientado hacia la izquierda
        recorte_horizontal = np.argmax(perfil_intensidad)
        imagen = imagen[:, recorte_horizontal:, :]
    # plt.figure()
    # plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    # plt.title('Recorte según Bordes')
    # plt.show()

    # Paso 1.3: Recorte adicional (25% derecho, 9% superior)
    filas, columnas, _ = imagen.shape
    recorte_derecho = int(columnas * 0.25)
    recorte_superior = int(filas * 0.09)
    imagen = imagen[recorte_superior:, :-recorte_derecho, :]
    # plt.figure()
    # plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    # plt.title('Recorte Adicional (25% derecho, 9% superior)')
    # plt.show()

    # Paso 1.4: Conversión a escala de grises
    if len(imagen.shape) == 3:  # Imagen RGB
        imagen_gris = np.mean(imagen, axis=2).astype(np.uint8)
    else:
        imagen_gris = imagen
    # plt.figure()
    # plt.imshow(imagen_gris, cmap='gray')
    # plt.title('Imagen en Escala de Grises')
    # plt.show()

    # Paso 1.5: Eliminación de áreas de interés
    alto_roi = 50
    ancho_roi = 140
    filas, columnas = imagen_gris.shape
    roi_superior = imagen_gris[:alto_roi, -ancho_roi:]
    if np.any(roi_superior):
        imagen_gris[:alto_roi, -ancho_roi:] = 0
    # plt.figure()
    # plt.imshow(imagen_gris, cmap='gray')
    # plt.title('Eliminación de Áreas de Interés')
    # plt.show()

    # Paso 2: Normalización
    imagen_gris_float = img_as_float(imagen_gris)

    # Paso 2.1: Filtro bilateral
    sigma_s = 2  # Desviación estándar espacial
    sigma_r = 0.2  # Desviación estándar de intensidad
    bordes = canny(imagen_gris_float)
    imagen_mediana = median(imagen_gris_float, disk(3))
    imagen_bilateral = cv2.bilateralFilter(imagen_mediana.astype(np.float32), d=-1, sigmaColor=sigma_r * 255, sigmaSpace=sigma_s)
    suavizado = np.copy(imagen_gris_float)
    suavizado[bordes] = imagen_bilateral[bordes]
    suavizado_uint8 = (suavizado * 255).astype(np.uint8)
    # plt.figure()
    # plt.imshow(suavizado_uint8, cmap='gray')
    # plt.title('Filtro Bilateral y Remoción de Ruido')
    # plt.show()

    

    # Paso 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clahe = cv2.createCLAHE(clipLimit=0.001, tileGridSize=(8, 8))
    # clahe_result = clahe.apply(suavizado_uint8)
    # plt.figure()
    # plt.imshow(clahe_result, cmap='gray')
    # plt.title('CLAHE (Mejora de Contraste)')
    # plt.show()

    return suavizado_uint8  



# Función contorno_to_mask
def contorno_to_mask(img, contorno, dilatado=0):
    mask = np.zeros_like(img)

    X = np.array([c[0][0] for c in contorno])
    Y = np.array([c[0][1] for c in contorno])

    X_min = min(X)
    X_max = max(X)
    Y_min, Y_max = min(Y), max(Y)

    for j in set(Y):
        vals_x = X[Y == j]
        mask[j, X_min - dilatado:max(vals_x) + dilatado] = 1
    mask = mask[Y_min:Y_max, X_min:X_max]
    img = img[Y_min:Y_max, X_min:X_max]
    img[mask != 1] = 0
    return img, mask

# Función clean
def clean(img, size=3, umbral=0.5):
    mask = np.zeros_like(img)
    img = np.array(img)

    y, x = img.shape
    X = np.arange(0, x - size, size)
    Y = np.arange(0, y - size, size)

    TX, TY = np.meshgrid(X, Y)
    TX = TX.flatten()
    TY = TY.flatten()
    for i, j in zip(TX, TY):
        i2 = i + size
        j2 = j + size
        box = img[j:j2, i:i2]
        val = np.mean(box)
        if val > umbral:
            mask[j:j2, i:i2] = 1.0
    return mask

# Función max_contorno
def max_contorno(contours):
    max_contour = []
    for contorno in contours:
        if len(max_contour) < len(contorno):
            max_contour = contorno
    return max_contour

# Función preprocesing
def preprocesing(img_original, vista='MLO', show=False):
    margen = 5
    img = np.array(img_original)
    img[:, :margen] = 0
    img[:, -margen:] = 0
    img[:margen, :] = 0
    img[-margen:, :] = 0

    y, x = img.shape
    sum_x = np.sum(img, axis=0)
    sum_x = [np.sum(sum_x[:int(x / 2)]), np.sum(sum_x[int(x / 2):])]
    if sum_x.index(max(sum_x)) == 1:
        img = cv2.flip(img, 1)

    _, thresh = cv2.threshold(img, 5, 1, cv2.THRESH_BINARY_INV)
    thresh = abs(thresh - 1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contorno = max_contorno(contours)
    img_mama, mask_mama = contorno_to_mask(img, contorno)

    umbral = np.mean(img_mama[mask_mama == 1]) * 1.2
    _, thresh_muscle = cv2.threshold(img_mama, umbral, 1, cv2.THRESH_BINARY_INV)
    thresh_muscle_aprox = abs(thresh_muscle - 1.0)
    mask_muscle_aprox = clean(thresh_muscle_aprox, size=3, umbral=0.85)
    kernel = np.ones((3, 3), np.uint8)
    mask_muscle_aprox = cv2.dilate(mask_muscle_aprox.astype(np.uint8), kernel, iterations=1)

    img_no_muscle = np.array(img_mama)
    contours, _ = cv2.findContours(mask_muscle_aprox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    Final_Mask = np.array(mask_mama)

    if show:
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img_mama, cmap='gray')

    espacio = 5
    for cont in contours:
        if np.sum(np.min(cont, axis=0)) == 0 and vista == 'MLO':
            c_x, c_y = [], []
            for c in cont:
                p_x = c[0][0]
                p_y = c[0][1]
                if p_x > espacio and p_y > espacio:
                    c_x.append(p_x)
                    c_y.append(p_y)

            y_max = np.max(c_y) + 10
            model = np.poly1d(np.polyfit(c_y, c_x, 3))

            yy = 0
            while True:
                xx = round(model(yy))
                if xx < 0 or xx >= x or yy > y_max:
                    break
                img_no_muscle[yy, :xx] = 0
                Final_Mask[yy, :xx] = 0
                yy += 1

    if show:
        plt.subplot(1, 3, 2)
        plt.imshow(img_no_muscle, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.imshow(Final_Mask, cmap='gray')
        plt.show()
        
    Final_Mask = Final_Mask.astype(float)
    Final_Mask[Final_Mask == 0] = np.nan
    img_no_muscle=img_no_muscle*Final_Mask
    return img_no_muscle, Final_Mask

def PDI_TOTAL(imagen):
    imagen_procesada = procesar_mamografia(imagen)
    img_no_muscle, Final_Mask = preprocesing(imagen_procesada, vista='CC', show=False)
    return img_no_muscle
    
# ruta_imagen = "133167, 01.07.13, RMLO.png"
# imagen=cv2.imread(ruta_imagen)
# imagen_procesada = procesar_mamografia(imagen)

# import os
# def procesar_carpeta_y_guardar_resultados(carpeta_base):
#     carpeta_salida = os.path.join(carpeta_base, "Resultados")
#     os.makedirs(carpeta_salida, exist_ok=True)

#     for archivo in os.listdir(carpeta_base):
#         if archivo.lower().endswith('.png'):
#             ruta_entrada = os.path.join(carpeta_base, archivo)
            
#             try:
#                 imagen = cv2.imread(ruta_entrada)
#                 imagen_salida = PDI_TOTAL(imagen)

#                 # Normalizar si no está en uint8
#                 if imagen_salida.dtype != 'uint8':
#                     imagen_salida = cv2.normalize(imagen_salida, None, 0, 255, cv2.NORM_MINMAX)
#                     imagen_salida = imagen_salida.astype('uint8')

#                 ruta_guardado = os.path.join(carpeta_salida, archivo)
#                 cv2.imwrite(ruta_guardado, imagen_salida)
#                 print(f"Guardada: {ruta_guardado}")
#             except Exception as e:
#                 print(f"Error procesando {archivo}: {e}")

# # Ejecutar la función con tu ruta
# carpeta_base = r"C:\Users\yakit\OneDrive\Escritorio\TT\Aplicacion\Sanos"
# procesar_carpeta_y_guardar_resultados(carpeta_base)


# # Aplicar el preprocesamiento a la imagen procesada
# img_no_muscle, Final_Mask = preprocesing(imagen_procesada, vista='MLO', show=True)

# # Mostrar resultados finales
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Imagen sin Músculo")
# plt.imshow(img_no_muscle, cmap='gray')

# plt.subplot(1, 2, 2)
# plt.title("Máscara Final")
# plt.imshow(Final_Mask, cmap='gray')
# plt.show()
