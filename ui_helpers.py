# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:53:46 2025

@author: Braulio
"""
# ui_helpers.py
import streamlit as st

# EM space for indentation
EM = "\u2003\u2003"

def sidebar_index(activo: str = "") -> None:
    """
    Índice lateral. Valores posibles de 'activo':
      'base', 'hurst', 'entropia', 'tda', 'diagnostico'
    """
    st.sidebar.markdown("## Índice")

    # 1. Cálculos
    st.sidebar.markdown(f"**1. Cálculos**{' ←' if activo=='base' else ''}")
    st.sidebar.page_link(
        "01_Calculos.py",
        label=f"{EM}Calcular dataset{' ←' if activo=='base' else ''}"
    )

    # 2. Resultados
    is_result = activo in {"hurst", "entropia", "tda"}
    st.sidebar.markdown(f"**2. Resultados**{' ←' if is_result else ''}")
    st.sidebar.page_link(
        "pages/02_CalculoHurst.py",
        label=f"{EM}Hurst{' ←' if activo=='hurst' else ''}"
    )
    st.sidebar.page_link(
        "pages/03_CalculoEntropia.py",
        label=f"{EM}Entropía{' ←' if activo=='entropia' else ''}"
    )
    st.sidebar.page_link(
        "pages/04_CalculoTDA.py",
        label=f"{EM}TDA{' ←' if activo=='tda' else ''}"
    )

    # 3. Clasificacion
    st.sidebar.markdown(f"**3. Diagnóstico**{' ←' if activo=='diagnostico' else ''}")
    st.sidebar.page_link(
        "pages/05_Clasificacion.py",
        label=f"{EM}Clasificación{' ←' if activo=='diagnostico' else ''}"
    )
