# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:32:46 2025
Manipulacion de carpetas
@author: ERICK
"""
import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import os
import shutil
import cv2
import pickle
import numpy as np
import PDI
import Hurst
import ENTROPIARENYI
import TDA as tda
def mover_archivos(origen, destino, extension): #mueve los archivos de una caprte con subcarpetas, a otra carpeta
    # Crear la carpeta de destino si no existe
    if not os.path.exists(destino):
        os.makedirs(destino)
    
    # Recorrer todas las subcarpetas de la carpeta origen
    for carpeta_raiz, _, archivos in os.walk(origen):
        for archivo in archivos:
            if archivo.lower().endswith(extension):
                origen_archivo = os.path.join(carpeta_raiz, archivo)
                destino_archivo = os.path.join(destino, archivo)

                # Evitar sobrescribir archivos con el mismo nombre
                if os.path.exists(destino_archivo):
                    base, ext = os.path.splitext(archivo)
                    contador = 1
                    while os.path.exists(destino_archivo):
                        destino_archivo = os.path.join(destino, f"{base}_{contador}{ext}")
                        contador += 1

                # Mover el archivo al destino
                shutil.move(origen_archivo, destino_archivo)
                print(f'Movido: {origen_archivo} -> {destino_archivo}')
# # Ejemplo de uso
# origen = "132717"
# destino = "Con_anomalia"
# extension = ".png"  # Cambiar por la extensión deseada
# mover_archivos(origen, destino, extension)

######################################################################################################
def aplicar_funcion_a_imagenes(carpeta_origen, carpeta_destino, funcion, nombre_personalizado,terminacion): #busaca el nombre de las imagenes, las carga y las pasa por la funcion
    if not os.path.exists(carpeta_destino):#crea destino si no existe                                        especificada
        os.makedirs(carpeta_destino)
        
    for carpeta_raiz, _, archivos in os.walk(carpeta_origen):
        for archivo in archivos:
            ruta_archivo = os.path.join(carpeta_raiz, archivo)
            img=cv2.imread(ruta_archivo)
            resultado=funcion(img)
            ruta_destino = os.path.join(carpeta_destino, archivo)
            ruta_destino = os.path.splitext(ruta_destino)[0] + f"_{nombre_personalizado}.{terminacion}"  # Cambiar la extensión a .py
            if terminacion in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:  # Formatos de imagen
                if isinstance(resultado, tuple):  # Si la función retorna más de un valor
                    resultado = resultado[0]  # Usar la primera salida (ej. imagen principal)
                cv2.imwrite(ruta_destino, resultado)
            elif terminacion=="npy":
                np.save(ruta_destino, resultado)
            else:
                with open(ruta_destino, 'wb') as archivo_destino:
                   pickle.dump(resultado, archivo_destino)
                print(f"Archivo guardado: {ruta_destino}")
 # # ejemplo de uso
# carpeta_origen="E:\mamografias\hurst\hurst\Sin_clahe"
# carpeta_destino="tumores_y_tejidos"
# carpetas.aplicar_funcion_a_imagenes(carpeta_origen, carpeta_destino, Hurst.aislarTumor, "separacion_tumor","npy")
def aplicar_funcion_a_imagenes_ventana(carpeta_origen, carpeta_destino, funcion,ventana, nombre_personalizado,terminacion): #busaca el nombre de las imagenes, las carga y las pasa por la funcion
    if not os.path.exists(carpeta_destino):#crea destino si no existe                                        especificada
        os.makedirs(carpeta_destino)
        
    for carpeta_raiz, _, archivos in os.walk(carpeta_origen):
        for archivo in archivos:
            ruta_archivo = os.path.join(carpeta_raiz, archivo)
            img=cv2.imread(ruta_archivo)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resultado=funcion(img,ventana)
            ruta_destino = os.path.join(carpeta_destino, archivo)
            ruta_destino = os.path.splitext(ruta_destino)[0] + f"_{nombre_personalizado}.{terminacion}"  # Cambiar la extensión a .py
            if terminacion in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:  # Formatos de imagen
                if isinstance(resultado, tuple):  # Si la función retorna más de un valor
                    resultado = resultado[0]  # Usar la primera salida (ej. imagen principal)
                cv2.imwrite(ruta_destino, resultado)
            elif terminacion=="npy":
                np.save(ruta_destino, resultado)
            else:
                with open(ruta_destino, 'wb') as archivo_destino:
                   pickle.dump(resultado, archivo_destino)
                print(f"Archivo guardado: {ruta_destino}")
###############################################################################################################
def cargar_archivos_pkl_a_diccionario(carpeta_origen):
    diccionario_datos = {}
    for archivo in os.listdir(carpeta_origen):
        if archivo.endswith(".pkl"):  # Filtrar solo archivos .pkl
            ruta_archivo = os.path.join(carpeta_origen, archivo)
            
            # Cargar el objeto desde el archivo .pkl
            with open(ruta_archivo, "rb") as archivo_pkl:
                objeto_cargado = pickle.load(archivo_pkl)
            
            # Usar el nombre del archivo (sin la extensión .pkl) como clave
            nombre_clave = os.path.splitext(archivo)[0]
            
            # Guardar el objeto cargado en el diccionario
            diccionario_datos[nombre_clave] = objeto_cargado
    
    return diccionario_datos
#diccionario_tumores_tejidos=cargar_archivos_pkl_a_diccionario("tumores_y_tejidos")
###############################################################################################################
def cargar_archivos_pkl(carpeta_origen):
    
    for archivo in os.listdir(carpeta_origen):
        if archivo.endswith(".pkl"):  # Filtrar solo archivos .pkl
            ruta_archivo = os.path.join(carpeta_origen, archivo)
            
            # Cargar el objeto desde el archivo .pkl
            with open(ruta_archivo, "rb") as archivo_pkl:
                objeto_cargado = pickle.load(archivo_pkl)
            
            # Usar el nombre del archivo (sin la extensión .pkl) como clave
            nombre_clave = os.path.splitext(archivo)[0]
            
            # Guardar el objeto cargado en el diccionario
    
    return objeto_cargado
###################################################################################################################        
def guardar_diccionario_como_pkl(diccionario, carpeta_destino):
    # Crear la carpeta si no existe
    os.makedirs(carpeta_destino, exist_ok=True)

    for clave, valor in diccionario.items():
        # Definir la ruta del archivo, usando la clave del diccionario como nombre
        ruta_archivo = os.path.join(carpeta_destino, f"{clave}.pkl")
        
        # Guardar el valor en un archivo .pkl
        with open(ruta_archivo, "wb") as archivo_pkl:
            pickle.dump(valor, archivo_pkl)
        
        print(f"Guardado: {ruta_archivo}")
###########################################################################################################
def procesar_lista_por_ventanas_guardar_pkl(lista, carpeta_destino, funcion, tVentana, proceso):
    # Crear la carpeta si no existe
    os.makedirs(carpeta_destino, exist_ok=True)
    indice=7
    for entrada in lista:
        resultado=funcion(entrada[1][0:200,0:200],tVentana)
        ruta_destino = os.path.join(carpeta_destino, f"{proceso}{indice}.pkl" )
        with open(ruta_destino, 'wb') as archivo_destino:
           pickle.dump(resultado, archivo_destino)
        print(f"Archivo guardado: {ruta_destino}")
        indice+=1
##############################################################################################################        
def procesar_lista_guardar_pkl(lista, carpeta_destino, funcion, proceso):
    # Crear la carpeta si no existe
    os.makedirs(carpeta_destino, exist_ok=True)
    indice=0
    for entrada in lista:
        resultado=funcion(entrada,8,0)
        ruta_destino = os.path.join(carpeta_destino, f"{proceso}{indice}.pkl" )
        with open(ruta_destino, 'wb') as archivo_destino:
           pickle.dump(resultado, archivo_destino)
        print(f"Archivo guardado: {ruta_destino}")
        indice+=1  
##############################################################################################################        
def procesar_lista(lista, carpeta_destino, funcion, proceso):
    # Crear la carpeta si no existe
    os.makedirs(carpeta_destino, exist_ok=True)
    lista_nueva=[]
    for entrada in lista:
        lista_nueva.append(funcion(entrada,8,0))        
    ruta_destino = os.path.join(carpeta_destino, f"{proceso}.pkl" )
    with open(ruta_destino, 'wb') as archivo_destino:
       pickle.dump(lista_nueva, archivo_destino)
    print(f"Archivo guardado: {ruta_destino}")    
        #indice+=1    
#########################################################################################################
def guardar_lista_pkl(lista, carpeta, nombre_archivo):
    """Guarda una lista en un archivo .pkl dentro de la carpeta especificada."""
    
    # Asegurar que la carpeta existe
    os.makedirs(carpeta, exist_ok=True)

    # Ruta completa del archivo
    ruta_completa = os.path.join(carpeta, nombre_archivo + ".pkl")

    # Guardar la lista como pickle
    with open(ruta_completa, "wb") as archivo:
        pickle.dump(lista, archivo)

    print(f" Lista guardada en: {ruta_completa}")   
    
 ##########################################################################################################   
def aislar_lista(Diccionario, multidimension=0, dato=0, guardar_vector=0):
    vector_de_mapas=[tupla for tupla in Diccionario.values()]
    lista_aislada=[]
    if multidimension==1:   
        for i in vector_de_mapas:
            lista_aislada.append(i[dato])
    else:
        for i in vector_de_mapas:
            lista_aislada.append(i)
    if guardar_vector==1:
        return lista_aislada
######################################################################################################
def procesar_archivos_en_carpeta(carpeta, funcion, tventana,proceso):
    # Recorre todos los archivos en la carpeta
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.txt'):
            # Construir la ruta completa del archivo
            ruta_txt = os.path.join(carpeta, archivo)
            
            # Paso 1: Cargar los datos desde el archivo .txt
            data = np.loadtxt(ruta_txt)
            
            # Paso 2: Aplicar la función a los datos
            if tventana!=0:
                resultados = funcion(data, tventana)
            else:
                resultados = funcion(data)
            # Paso 3: Guardar los resultados en un archivo .pkl
            archivo=proceso+archivo
            ruta_pkl = os.path.join(carpeta, archivo.replace('.txt', '.pkl'))
            with open(ruta_pkl, 'wb') as f:
                pickle.dump(resultados, f)
            
            print(f"Resultados guardados en: {ruta_pkl}")
            
def cargar_txt_lista(carpeta):
    # Recorre todos los archivos en la carpeta
    lista=[]
    for archivo in os.listdir(carpeta):
        listaaux=[]
        if archivo.endswith('.txt'):
            # Construir la ruta completa del archivo
            ruta_txt = os.path.join(carpeta, archivo)
            
            # Paso 1: Cargar los datos desde el archivo .txt
            data = np.loadtxt(ruta_txt)
            listaaux.append(ruta_txt)
            
            listaaux.append(data)
            lista.append(listaaux)
    return lista
            
def procesar_archivos_en_listaNombres(lista, funcion, tventana, proceso,carpeta):
    # Recorre todos los archivos en la carpeta
    for i in lista:
        
        # Construir la ruta completa del archivo
        ruta_txt = i[0]+proceso+".pkl"
        
        # Paso 1: Cargar los datos desde el archivo .txt
        data = i[1]
        
        # Paso 2: Aplicar la función a los datos
        data=data[0:200,0:200]
        
        resultados = funcion(data, tventana)
        
        # Paso 3: Guardar los resultados en un archivo .pkl
        ruta_pkl = os.path.join(carpeta, ruta_txt)
        with open(ruta_pkl, 'wb') as f:
            pickle.dump(resultados, f)
        
        print(f"Resultados guardados en: {ruta_pkl}")
        
################################## PROCESOS PARA BASES DE DATOS ########################################


def cargar_imagenes_grises_en_dataframe():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    rutas, _ = QFileDialog.getOpenFileNames(
        None,
        "Selecciona múltiples imágenes",
        "",
        "Imágenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    )

    if not rutas:
        print("No se seleccionaron imágenes.")
        return None

    datos = []

    for ruta in rutas:
        print(f"Cargando: {ruta}")
        img_gray = PDI.PDI_TOTAL(cv2.imread(ruta))
        if img_gray is None:
            print(f"Error al leer la imagen: {ruta}")
            continue
        alto, ancho = img_gray.shape
        datos.append({
            "ruta": ruta,
            "Imagen": img_gray,  # array 2D de intensidades
            "dimensiones": (alto, ancho)
        })               
    df = pd.DataFrame(datos)
    return df

def agregar_Hurst(datos):
    columnas_resultantes = [
        'Delta Alpha', 'Delta Falpha', 'Alpha Estrella', 'Asimetria',
        'Curtosis', 'Curvatura', 'Alphas', 'F Alphas', 'Mapa'
    ]

    resultados = {col: [] for col in columnas_resultantes}

    for imagen in datos['Imagen']:
        salida = Hurst.datos_a_partir_de_imagen_PDI(imagen)
        for col, val in zip(columnas_resultantes, salida):
            resultados[col].append(val)

    for col in columnas_resultantes:
        datos[col] = resultados[col]

    return datos
    
def agregar_entropia(datos):
    columnas=['Entropia Renyi', 'Complejidad Renyi','S']
    resultados = {col: [] for col in columnas}

    for imagen in datos['Imagen']:
        salida = ENTROPIARENYI.obtener_entropia_y_complejidad_Renyi(imagen)
        for col, val in zip(columnas, salida):
            resultados[col].append(val)

    for col in columnas:
        datos[col] = resultados[col]

    return datos

def agregar_TDA(datos):
    columnas=['nac_h0', 'mur_h0', 'lif0', 'pe_h0', 'sh_h0','nac_h1', 'mur_h1', 'lif1', 'pe_h1', 'sh_h1']
    resultados = {col: [] for col in columnas}

    for imagen in datos['Imagen']:
        salida = tda.procesar_imagen_topologica(imagen)
        for col, val in zip(columnas, salida):
            resultados[col].append(val)

    for col in columnas:
        datos[col] = resultados[col]

    return datos
    
def calcular_todo(datos):
    datos=agregar_Hurst(datos)
    datos=agregar_entropia(datos)
    datos=agregar_TDA(datos)
    return datos





