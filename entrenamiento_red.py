# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 17:11:38 2025

@author: ERICK
"""
# entrenar_guardar_xgboost.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ===========================
# 1. Cargar datos
# ===========================
df = pd.read_csv("protobase_chida.csv")

# Columnas seleccionadas
columnas_interes = [
    'Delta Alpha', 'Delta Falpha', 'Asimetria', 'Curtosis', 'Std_Entr',
    'Std_Comp', 'S','H0_EntropiaPersistencia', 'H1_EntropiaPersistencia'
]

X = df[columnas_interes]
y = df['Resultados']

# ===========================
# 2. División de datos
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# ===========================
# 3. Entrenar modelo
# ===========================
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric=['logloss', 'error'],
    random_state=42
)

eval_set = [(X_train, y_train), (X_test, y_test)]

model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False
)

# ===========================
# 4. Evaluación básica
# ===========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Evaluación del Modelo ===")
print(f"Accuracy en test: {accuracy:.4f}")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred))

# ===========================
# 5. Guardar modelo
# ===========================
joblib.dump(model, "modelo_xgboost.pkl")
print("\n✅ Modelo guardado como 'modelo_xgboost.pkl'")

# ===========================
# 6. Gráficas
# ===========================
evals_result = model.evals_result()
epochs = len(evals_result['validation_0']['logloss'])

# 6a. Log Loss
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), evals_result['validation_0']['logloss'], label='Entrenamiento')
plt.plot(range(epochs), evals_result['validation_1']['logloss'], label='Validación')
plt.xlabel('Iteraciones')
plt.ylabel('Log Loss')
plt.title('Log Loss durante el entrenamiento')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6b. Accuracy
train_acc = [1 - e for e in evals_result['validation_0']['error']]
val_acc = [1 - e for e in evals_result['validation_1']['error']]

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), train_acc, label='Entrenamiento')
plt.plot(range(epochs), val_acc, label='Validación')
plt.xlabel('Iteraciones')
plt.ylabel('Accuracy')
plt.title('Accuracy durante el entrenamiento')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6c. Importancia de características
plt.figure(figsize=(8, 6))
plot_importance(model, importance_type='gain', max_num_features=10)
plt.title("Importancia de características (Gain)")
plt.tight_layout()
plt.show()

# # 6d. Visualizar el primer árbol
# plt.figure(figsize=(30, 15))
# plot_tree(model, num_trees=0, rankdir='LR')
# plt.title("Visualización del primer árbol")
# plt.tight_layout()
# plt.show()

# ===========================
# 7. Número de árboles
# ===========================
print(f"\n🌳 El modelo contiene {len(model.get_booster().get_dump())} árboles.")
