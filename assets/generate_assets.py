
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

# Configuración de rutas
BASE_DIR = r"c:\Users\eliut\OneDrive\Escritorio\hola, respaldo del respaldo jajaja\Cuatrimestre_II\Aprendizaje automático\Predict-Heart"
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

def safe_load(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

# Cargar modelos y datos
modelos_raw = {
    'Regresión Logística': 'modelo_LogisticRegression_Base.pkl',
    'Naive Bayes': 'modelo_NaiveBayes_Base.pkl',
    'KNN': 'modelo_KNN_Optimizado.pkl',
    'Árbol de Decisión': 'modelo_DecisionTree_Optimizado.pkl',
    'SVM': 'modelo_SVM_Optimizado.pkl',
    'Red Neuronal MLP': 'modelo_MLP_Optimizado.pkl'
}

modelos = {}
for nombre, filename in modelos_raw.items():
    m = safe_load(filename)
    if m: modelos[nombre] = m

X_test = safe_load('X_test_scaled.pkl')
y_test = safe_load('y_test.pkl')

if X_test is None or y_test is None:
    print("Error: No se pudieron cargar los datos de test.")
    exit()

# 1. Generar Curva ROC combinada
plt.figure(figsize=(10, 7))
for nombre, modelo in modelos.items():
    if hasattr(modelo, "predict_proba"):
        y_score = modelo.predict_proba(X_test)[:, 1]
    elif hasattr(modelo, "decision_function"):
        y_score = modelo.decision_function(X_test)
    else:
        continue
    
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{nombre} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curvas ROC - Comparativa de Modelos')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(ASSETS_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Generar Matrices de Confusión (Top 3 modelos por Accuracy)
from sklearn.metrics import accuracy_score
accuracies = {n: accuracy_score(y_test, m.predict(X_test)) for n, m in modelos.items()}
top_3 = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)[:3]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (nombre, acc) in enumerate(top_3):
    cm = confusion_matrix(y_test, modelos[nombre].predict(X_test))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=axes[idx],
                xticklabels=["Negativo", "Positivo"], yticklabels=["Negativo", "Positivo"])
    axes[idx].set_title(f"{nombre}\nAccuracy: {acc:.2%}")

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Importancia de Características (para modelos que lo soportan)
# Como solo hay 3 características (Edad, CK-MB, Troponina)
features = ['Edad', 'CK-MB', 'Troponina']
# Usaremos el Árbol de Decisión como ejemplo
if 'Árbol de Decisión' in modelos:
    importancia = modelos['Árbol de Decisión'].feature_importances_
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importancia, y=features, palette='viridis')
    plt.title('Importancia de Características (Árbol de Decisión)')
    plt.xlabel('Importancia Relativa')
    plt.savefig(os.path.join(ASSETS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Assets generados en {ASSETS_DIR}")
