# -*- coding: utf-8 -*-
"""
Created on Thu May 29 15:09:00 2025

@author: Braulio
"""
# =============================================================================
#  ENTROPIA DE SHANNON NORMALIZADA DIVERGENCIA JEHNSEN SHANNON Y COMPLEJIDAD
# =============================================================================


import cv2
import numpy as np
from collections import Counter
from scipy.stats import entropy


def grises(imagen):
    """
    gris: Devuelve una imagen en escala de grises, con el metodo de luminancia 
    ponderada

    """
    im = cv2.imread(imagen)
    B,G,R = cv2.split(im)
    
    gris = 0.2125 * R + 0.07154 * G + 0.0721 * B
    
    gris= gris.astype(np.uint8)
    gris=gris.astype(float)
    gris[gris == 0] = np.nan
    
    return gris


def matriz_A_Patrones(A, dx, dy):
    
    fil = len(A)
    col = len(A[0])  
    patrones_Presentes = []
    #Para hacerlo con translape for i in range(fil - dx + 1)
    #Sin Translape for i in range(0,fil - dx + 1,dx):
    for i in range(fil - dx + 1):
        for j in range(col - dy + 1):
            submat = np.array([fila[j:j+dy] for fila in A[i:i+dx]])

            # Si hay NaN en la submatriz, saltamos esta iteración
            if np.isnan(submat).any():
                continue

            # Si NO hay NaN, procesamos la submatriz
            patrones_Presentes.append(extraer_Patrones(submat))


    return patrones_Presentes


def to_hashable(contador):
    """
    FUNCION AUXILAR PARA USAR COUNTER
    Convierte recursivamente listas en tuplas de tipo hasheable para usar tuplas
    """
    if isinstance(contador, list):
        return tuple(to_hashable(item) for item in contador)
    return contador

def frecuencia_Patrones_Unicos(patrones):
    # Convertir cada patrón (que es una lista) a su versión hashable
    patrones_hashables = [to_hashable(p) for p in patrones]
    """
    Al tener un objeto de tipo 'Counter' tienes keys y values 
    Keys corresponde a los patrones unicos
    Values corresponde a la frecuencia de cada patron unico
    Objeto_Counter.keys() --> Patrones Unicos en la imagen da tuplas
    Objeto_Counter.values() --> Valor de frecuencia de cada patron 
    """
    patronesUnicos_Frecuencias = Counter(patrones_hashables) #Objeto counter con sus keys y su frecuencia
    frecuencias = list(patronesUnicos_Frecuencias.values())
    patronesUnicos = list(patronesUnicos_Frecuencias.keys())
    return patronesUnicos,frecuencias

    
def extraer_Patrones(matriz):
    valores_aplanados = [elem for fila in matriz for elem in fila]
    valores_unicos_ordenados = sorted(set(valores_aplanados))
    
    mapeo = {}
    for i, valor in enumerate(valores_unicos_ordenados, start=1):
        mapeo[valor] = i
    
    nueva_matriz = []
    for fila in matriz:
        nueva_fila = [mapeo[val] for val in fila]
        nueva_matriz.append(nueva_fila)
    
    return nueva_matriz

# =============================================================================
# ENTROPIA DE RENYI
# =============================================================================
def Entropia(patrones,tipo='N'):
    """
   Calcula la entropía de Shannon de un conjunto de patrones.

   Parámetros:
   - patrones: Lista o arreglo con los patrones.
   - tipo: "S" para entropía sin normalizar, "N" para entropía normalizada (por defecto).

   Retorna:
   - Entropía calculada.
   - Vector de probabilidades P.
   """
    patrones_Unicos,frecuencias = frecuencia_Patrones_Unicos(patrones)
    frecuencias=np.array(frecuencias)
    
    #nf=np.arange(1,(len(frecuencias))+1)
    # print(len(frecuencias))

    # plt.figure(figsize=(8,5))
    # plt.bar(nf, np.log(frecuencias),color='#00FC26')
    # plt.title('Histograma de frecuencias')
    # plt.xlabel('Patrones unicos')
    # plt.ylabel('Frecuencia')
    # plt.show()
    suma_freq = np.sum(frecuencias)
    # print(suma_freq)
    P = frecuencias/suma_freq #VECTOR Probabilidades
    # print(np.sum(P))
    n=len(patrones_Unicos)
    # print(n)
    #ENTROPIA DE SHANON
    #Caso especial donde una distribucion uniforme es = 0
    if n <=1:
        return 0,P
    
    entropia=entropy(P)
    
    if tipo == "N":
        entropia /= np.log(n) #ENTROPIA DE SHANON NORMALIZADA
    return entropia,P


def entropia_Renyi(patrones,alphas):
    P=Entropia(patrones,tipo='S')[1]
    entropiasR=[]
    for alpha in alphas:
        
        if alpha == 1:
            R = -np.sum(P * np.log(P))
             
        else:
            suma_p_alpha = np.sum(P ** alpha)
            R = (1/(1 - alpha))*np.log(suma_p_alpha)
        entropiasR.append(R)
    return np.array(entropiasR)
        


def divergenciaRenyi(patrones,alphas):
    
    
    P=Entropia(patrones,tipo='S')[1]
    n=len(P)
    
    U = np.ones(n)/n #uniforme
    divergencias = []
    
    for alpha in alphas:
        if alpha == 1:
            #Kullbacklieber
            diver = np.sum(P * np.log(P / U))
        else:
            #Divergencia de Renyi
            diver = (1 / (alpha-1)) * np.log(np.sum((P ** alpha) / (U ** (alpha - 1))))
            
        divergencias.append(diver)
    return np.array(divergencias)

def complejidadRenyi(entropiaR,divergenciaR):
    
    return entropiaR * divergenciaR

def obtener_entropia_y_complejidad_Renyi(entrada, es_ruta=False):
    """
    Calcula la entropía de Rényi, la divergencia de Rényi y la complejidad de Rényi de una imagen o matriz.
    
    Parámetros:
    - entrada: Puede ser una ruta de imagen (str) o una matriz numpy.
    - alphas: Lista de valores de α para la entropía de Rényi.
    - es_ruta: Booleano que indica si 'entrada' es una ruta de archivo (True) o una matriz (False).

    Retorna:
    - entropías_Renyi: Lista de valores de entropía de Rényi.
    - complejidades_Renyi: Lista de valores de complejidad de Rényi.
    """
    gris = grises(entrada) if es_ruta else entrada
    
    alphas=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5]
    

    patrones = matriz_A_Patrones(gris, 3, 3)

    entropiaR = entropia_Renyi(patrones, alphas)
    divergenciaR = divergenciaRenyi(patrones, alphas)
    complejidadR = complejidadRenyi(entropiaR, divergenciaR)
    dx = np.diff(entropiaR)
    dy = np.diff(complejidadR)
    distancias = np.sqrt(dx**2 + dy**2)
    S=np.sum(distancias)
    return entropiaR, complejidadR,S



#Ejemplo de uso de funcion 
# Dimensiones de la matriz
# altura = 256
# ancho = 256

# # Ruido blanco: valores aleatorios en [0, 255] como enteros tipo uint8
# ruido_blanco = np.random.randint(0, 256, size=(altura, ancho), dtype=np.uint8)
# Variable=obtener_entropia_y_complejidad_Renyi(ruido_blanco)

import os
import pandas as pd


# # [Aquí pegas todas tus funciones auxiliares...]

def procesar_carpeta_y_guardar_csv(ruta_carpeta, nombre_csv):
    resultados = []
    for archivo in os.listdir(ruta_carpeta):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            ruta_imagen = os.path.join(ruta_carpeta, archivo)
            try:
                entropiaR, complejidadR, S = obtener_entropia_y_complejidad_Renyi(ruta_imagen, es_ruta=True)
                # Guarda los vectores como string estilo lista de Python
                resultados.append({
                    'id_imagen': archivo,
                    'entropia': str(list(entropiaR)),
                    'complejidad': str(list(complejidadR)),
                    'S': S
                })
            except Exception as e:
                print(f"Error procesando {archivo}: {e}")
    df = pd.DataFrame(resultados)
    df.to_csv(nombre_csv, index=False)
    print(f"Guardado {nombre_csv} con {len(resultados)} imágenes procesadas.")

# # ========== RUTAS Y SALIDAS ==========
# carpetas = [
#     (r"C:\Users\yakit\OneDrive\Escritorio\TT\Aplicacion\Experimento", "resultado_raro.csv"),
#     # (r"C:\Users\yakit\OneDrive\Escritorio\TT\Aplicacion\Enfermos", "resultados_Enfermos.csv")
# ]

# for ruta, archivo_salida in carpetas:
#     procesar_carpeta_y_guardar_csv(ruta, archivo_salida)
