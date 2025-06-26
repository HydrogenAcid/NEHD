# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:08:39 2025

@author: Braulio
"""
# api_calculo.py  ── versión mínima con nombres fijos
import os, threading, pickle
from flask import Flask, request, jsonify

# Carpeta donde vive este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nombres EXACTOS que ya usas
INPUT_PKL  = os.path.join(BASE_DIR, "input_para_el_calculo.pkl")
OUTPUT_PKL = os.path.join(BASE_DIR, "resultados_tda_entropia.pkl")

app = Flask(__name__)

# ─────────────────── HILO DE CÁLCULO ───────────────────
def _worker():
    """
    Llama a tu función main(INPUT_PKL, OUTPUT_PKL) en segundo plano.
    """
    from calculo_tda_entropia import main  # importa aquí para no bloquear

    try:
        main(INPUT_PKL, OUTPUT_PKL)
    except Exception as e:
        # Registra el error en un archivo para depurar si algo truena
        with open(os.path.join(BASE_DIR, "error_calculo.log"), "w") as f:
            f.write(str(e))

# ─────────────────── ENDPOINT ÚNICO ───────────────────
@app.route("/procesar", methods=["POST"])
def procesar():
    """
    Guarda el PKL recibido como 'input_para_el_calculo.pkl', lanza el hilo
    y responde 202 en menos de un segundo.
    """
    if "archivo" not in request.files:
        return jsonify({"error": "Falta el campo 'archivo'"}), 400

    # Sobrescribe (o crea) el archivo temporal con el DataFrame
    request.files["archivo"].save(INPUT_PKL)

    # Arranca el cálculo en segundo plano
    threading.Thread(target=_worker, daemon=True).start()

    # Respuesta inmediata
    return jsonify({"status": "started"}), 202

# ─────────────────── MAIN ───────────────────
if __name__ == "__main__":
    # 127.0.0.1 para local; cambia si lo expones fuera
    app.run(host="127.0.0.1", port=5005, debug=False)
