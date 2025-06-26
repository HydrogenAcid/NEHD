@echo off
REM Cambia esta ruta si tu instalación de conda está en otro lugar
call C:\Users\yakit\conda3\Scripts\activate.bat base

cd C:\Users\yakit\OneDrive\Escritorio\TT\Aplicacion\Main

REM Ejecuta streamlit usando python -m para asegurar que tome el entorno correcto
python -m streamlit run 01_Calculos.py

pause
