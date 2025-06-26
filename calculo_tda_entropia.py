# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:02:27 2025

@author: Braulio
"""
# calculo_tda_entropia.py

import pickle
import sys
import carpetas
import pandas as pd
import os

def main(input_pkl, output_pkl):
    print("Estoy corriendo en:", os.getcwd(), flush=True)
    with open(input_pkl, "rb") as f:
        df_imgs = pickle.load(f)

    # --- Cálculo Hurst ---
    cols_hurst = ['Delta Alpha','Delta Falpha','Alpha Estrella','Asimetria',
                  'Curtosis','Curvatura','Alphas','F Alphas','Mapa']
    for c in cols_hurst:
        if c not in df_imgs.columns:
            df_imgs[c] = None
    for i, img in enumerate(df_imgs["Imagen"]):
        salida = carpetas.Hurst.datos_a_partir_de_imagen_PDI(img)
        for c, v in zip(cols_hurst, salida):
            df_imgs.at[i, c] = v

    # --- Cálculo TDA ---
    cols_tda = ['nac_h0','mur_h0','lif0','pe_h0','sh_h0',
                'nac_h1','mur_h1','lif1','pe_h1','sh_h1']
    for c in cols_tda:
        if c not in df_imgs.columns:
            df_imgs[c] = None
    for i, img in enumerate(df_imgs["Imagen"]):
        salida = carpetas.tda.procesar_imagen_topologica(img)
        for c, v in zip(cols_tda, salida):
            df_imgs.at[i, c] = v

    # --- Cálculo Entropía ---
    cols_ent = ['Entropia Renyi','Complejidad Renyi','S']
    for c in cols_ent:
        if c not in df_imgs.columns:
            df_imgs[c] = None
    for i, img in enumerate(df_imgs["Imagen"]):
        salida = carpetas.ENTROPIARENYI.obtener_entropia_y_complejidad_Renyi(img)
        for c, v in zip(cols_ent, salida):
            df_imgs.at[i, c] = v

    # --- Guarda solo el PKL ---
    with open(output_pkl, "wb") as f:
        pickle.dump(df_imgs, f)

    print("¡Cálculo completo! PKL generado.")

# --- Soporte para llamada desde línea de comando O desde la API ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUso: python calculo_tda_entropia.py <input_pkl> <output_pkl>\n")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

