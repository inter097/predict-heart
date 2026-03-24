# Predicción de Riesgo de Ataque Cardíaco 🫀

Este proyecto utiliza técnicas de **Aprendizaje Automático (Machine Learning)** para predecir el riesgo de un ataque cardíaco en pacientes basándose en biomarcadores clínicos. La aplicación ha sido desarrollada con **Streamlit** para proporcionar una interfaz interactiva y fácil de usar para profesionales de la salud.

## 🚀 Características
- **Multimodelo**: Compara predicciones de 6 modelos diferentes (Regresión Logística, Naive Bayes, KNN, Árbol de Decisión, SVM y Redes Neuronales).
- **Visualización Interactiva**: Gráficos de probabilidad (Pie charts), curvas ROC y matrices de confusión.
- **Métricas de Evaluación**: Análisis detallado de Accuracy, Precision, Recall, F1-score, F0.5 y F2-score.
- **Análisis de Rendimiento**: Comparativa del tiempo de inferencia entre modelos.

## 📊 Biomarcadores Utilizados
La predicción se basa en tres factores clave:
1. **Edad**: Factor de riesgo cardiovascular acumulativo.
2. **CK-MB**: Enzima indicadora de daño en el músculo cardíaco.
3. **Troponina**: Proteína específica liberada durante una lesión cardíaca.

## 🛠️ Estructura del Proyecto
```text
├── app/                # Código fuente de la aplicación Streamlit
│   └── main.py
├── datasets/           # Conjuntos de datos utilizados
├── models/             # Modelos entrenados (.pkl) y escaladores
├── notebooks/          # Jupyter Notebooks (EDA y Modelado)
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación
```

## ⚙️ Instalación y Ejecución Local

1. **Clonar el repositorio**:
   ```bash
   git clone <url-del-repositorio>
   cd T10
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar la aplicación**:
   ```bash
   streamlit run app/main.py
   ```

## 🧠 Modelos Implementados
- **Regresión Logística**: Modelo base lineal.
- **Naive Bayes**: Clasificador probabilístico.
- **K-Nearest Neighbors (KNN)**: Optimizado para clasificación por cercanía.
- **Árbol de Decisión**: Modelo jerárquico optimizado.
- **Support Vector Machine (SVM)**: Clasificación mediante hiperplanos.
- **Red Neuronal (MLP)**: Perceptrón multicapa para captura de patrones complejos.

---
*Desarrollado como parte del Cuatrimestre II - Aprendizaje Automático.*
