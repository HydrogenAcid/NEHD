# -*- coding: utf-8 -*-
"""
Created on Thu May 29 15:55:06 2025

@author: Braulio
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PDI import PDI_TOTAL
import pickle
import time
import requests
from pathlib import Path
import datetime as dt
LOGO_PATH = "logo.jpg"

st.markdown(
    """
    <style>
      /* Oculta toda la lista de navegación automática */
      [data-testid="stSidebarNav"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

from ui_helpers import sidebar_index
sidebar_index(activo="base")

def lanzar_tarea(df, timeout=60):
    """
    Serializa el DataFrame y lo envía al endpoint /procesar de tu API.
    Lanza excepción si hay error.
    """
    buf = io.BytesIO()
    pickle.dump(df, buf)
    buf.seek(0)
    resp = requests.post(
        f"{API_URL}/procesar",
        files={"archivo": ("input_para_el_calculo.pkl", buf.getvalue())},
        timeout=timeout
    )
    resp.raise_for_status()
def cargar_imagenes_grises_en_dataframe_streamlit(uploaded_files):
    datos = []
    for archivo in uploaded_files:
        bytes_imagen = archivo.read()
        np_arr = np.frombuffer(bytes_imagen, np.uint8)
        imagen_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if imagen_color is None:
            st.warning(f"No se pudo leer la imagen: {archivo.name}")
            continue

        try:
            imagen_procesada = PDI_TOTAL(imagen_color)
            if imagen_procesada is None:
                st.warning(f"No se pudo procesar la imagen: {archivo.name}")
                continue

            alto, ancho = imagen_procesada.shape
            datos.append({
                "ruta": archivo.name,
                "Imagen": imagen_procesada,
                "dimensiones": (alto, ancho)
            })
        except Exception as e:
            st.error(f"Error procesando {archivo.name}: {e}")
            continue

    if datos:
        return pd.DataFrame(datos)
    return None

# ---- Estado global ----
if "df_imgs" not in st.session_state:
    st.session_state.df_imgs = None
if "entropia_done" not in st.session_state:
    st.session_state.entropia_done = False

        
col1, col2 = st.columns([1, 8])
with col1:
    if os.path.exists(LOGO_PATH):
        logo = Image.open(LOGO_PATH)
        st.image(logo, width=80)
    else:
        st.write("⠀")
with col2:
    st.markdown("""
        <h1 style='color: white; margin: 0; padding: 0; font-family: "Times New Roman"; font-size: 36pt; text-align: center;'>
            N E H D &lt;{•.•}&gt;
        </h1>
        <hr style='border-color: white; margin-top: 4px; margin-bottom: 12px;'/>
    """, unsafe_allow_html=True)

st.markdown("""
    <p style='text-align: center; color: white; font-family: "Times New Roman"; font-size: 18pt; margin-top: 0; margin-bottom: 20px;'>
       Bienvenido carga tus imágenes para procesarlas automáticamente.
    </p>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Selecciona imágenes",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True
)

if uploaded_files and st.session_state.df_imgs is None:          # ← aquí el cambio
    with st.spinner("Procesando imágenes..."):
        df_imgs = cargar_imagenes_grises_en_dataframe_streamlit(uploaded_files)
        if df_imgs is not None:
            st.session_state.df_imgs = df_imgs
            st.success(f"Se cargaron {len(df_imgs)} imágenes.")
        else:
            st.error("No se pudieron cargar imágenes válidas.")

# Vista previa
if st.session_state.df_imgs is not None and not st.session_state.df_imgs.empty:
    df_imgs = st.session_state.df_imgs

    st.markdown(f"<b style='color:white;'>Imágenes cargadas: {len(df_imgs)}</b>", unsafe_allow_html=True)
    img_idx = st.number_input("Navega entre imágenes cargadas", min_value=0, max_value=len(df_imgs)-1, value=0, step=1, key="img_selector")
    st.caption(f"Imagen {img_idx+1} de {len(df_imgs)}")

    img_np = df_imgs.iloc[img_idx]["Imagen"]
    if img_np.dtype != np.uint8:
        if img_np.ptp() > 0:
            img_np = (255 * (img_np - img_np.min()) / img_np.ptp()).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    try:
        img_pil = Image.fromarray(img_np, mode='L')

        def pil_to_centered_html(pil_img):
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            import base64
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            html = f"""
            <div style="text-align: center; margin: 10px 0;">
                <img src="data:image/png;base64,{b64}" style="max-width:600px; height:auto;"/>
            </div>
            """
            return html

        st.markdown("<h3 style='color:white; text-align:center; font-family:Times New Roman; font-size:24pt;'>Vista previa de imagen procesada</h3>", unsafe_allow_html=True)
        st.markdown(pil_to_centered_html(img_pil), unsafe_allow_html=True)
        st.caption(f"Imagen procesada: {df_imgs.iloc[img_idx]['ruta']} (dim: {df_imgs.iloc[img_idx]['dimensiones']})")
        # ========================
# = VISTA DEL DATAFRAME =
# ========================
        # st.markdown("<h3 style='color:white; text-align:center;'>Tabla de imágenes procesadas</h3>", unsafe_allow_html=True)
        # df_display = df_imgs.copy()
        # df_display["Imagen"] = df_display["Imagen"].apply(lambda x: f"array{tuple(x.shape)}, dtype={x.dtype}")
        # st.dataframe(df_display)
        

# Termina el bloque try de la visualización de imágenes...
    except Exception as e:
        st.error(f"Error mostrando la imagen procesada: {e}")

# ─────────── CONFIG ───────────
BASE_DIR   = Path(__file__).parent
RESULT_PKL = BASE_DIR / "resultados_tda_entropia.pkl"

API_URL      = "http://localhost:5005"
CHECK_EVERY  = 300                     # 5 min = 300 s
MAX_WAIT_SEC = 12 * 60 * 60            # 12 h

# ─────────── ESTILOS GLOBALES (botón centrado + alertas) ───────────
st.markdown(
    """
    <style>
    /* 1️⃣  Centrar y agrandar TODOS los botones st.button */
    div[data-testid="stButton"] {
        display: flex;              /* usa flexbox */
        justify-content: center;    /* centra horizontal */
    }
    div[data-testid="stButton"] > button {
        font-size: 1.4rem;          /* más grande */
        padding: 1rem 3.2rem;       /* más ancho/alto */
    }

    /* 2️⃣  Alertas (success, info, error, warning) más vistosas */
    div[data-testid="stAlert"] {
        border-radius: 0.75rem;
        padding: 1.2rem;
        font-size: 1.15rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        animation: fadeIn 0.4s ease-in-out;
        max-width: 830px;
        margin: 0.5rem auto;        /* centradas */
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-6px);}
        to   {opacity: 1; transform: translateY(0);}
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ─────────── FUNC. UTIL ───────────
def lanzar_tarea(df):
    buf = io.BytesIO()
    pickle.dump(df, buf)
    buf.seek(0)
    requests.post(
        f"{API_URL}/procesar",
        files={"archivo": ("input_para_el_calculo.pkl", buf.getvalue())},
        timeout=10
    ).raise_for_status()

# ─────────── UI ───────────

# Flags en session_state
if "esperando"   not in st.session_state: st.session_state.esperando = False
if "inicio_esp"  not in st.session_state: st.session_state.inicio_esp = None

# --- Caso 0: el PKL ya existía desde antes ---
if RESULT_PKL.exists() and not st.session_state.esperando:
    st.success("✅ Cálculo hecho previamente. Puede pasar a la ventana de resultados.")
    if "resultado_df" not in st.session_state:
        with RESULT_PKL.open("rb") as f:
            st.session_state["resultado_df"] = pickle.load(f)
    # Botón/link a resultados
    st.page_link("pages/02_CalculoHurst.py", label="Ir a Resultados ➡️")

# --- Botón para lanzar cálculo ---
elif (
    st.session_state.get("df_imgs") is not None
    and not st.session_state.esperando
):
    if st.button("🚀  Calcular todo", key="btn_calc"):
        try:
            lanzar_tarea(st.session_state.df_imgs)
            st.session_state.esperando   = True
            st.session_state.inicio_esp = dt.datetime.now()
            st.success("Tarea iniciada. La página se actualizará automáticamente.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ No se pudo arrancar el cálculo:\n{e}")

# --- Seguimiento: vigilar archivo cada 5 min, máximo 12 h ---
if st.session_state.esperando:
    if RESULT_PKL.exists():
        st.success("✅ Cálculo exitoso. Puede pasar a la ventana de resultados.")
        st.session_state.esperando = False
        with RESULT_PKL.open("rb") as f:
            st.session_state["resultado_df"] = pickle.load(f)
        st.page_link("pages/02_CalculoHurst.py", label="Ir a Resultados ➡️")

    else:
        # tiempo transcurrido
        trans = (dt.datetime.now() - st.session_state.inicio_esp).total_seconds()
        if trans > MAX_WAIT_SEC:
            st.error("⏰ Tiempo agotado (12 h). El cálculo no terminó.")
            st.session_state.esperando = False
        else:
            mins_left = int((MAX_WAIT_SEC - trans) / 60)
            st.info(
                f"Calculando… se volverá a comprobar en {CHECK_EVERY//60} min. "
                f"(Tiempo restante máximo: {mins_left} min)"
            )
            time.sleep(CHECK_EVERY)
            st.rerun()