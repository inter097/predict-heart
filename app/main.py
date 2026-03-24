import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import os
import sys


# Cargar modelos
import joblib

import joblib

# Configuración de rutas (relativas al archivo main.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Función para cargar archivos de forma segura
def safe_load(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"No se encontró el archivo, verifique la ruta: {filename}")
        return None

# Inicializar
modelos = {}
scaler = None
X_test = None
y_test = None
mostrar_matriz = False


# Bloque 1: Cargar modelos
modelos_raw = {
    'Regresión Logística': 'modelo_LogisticRegression_Base.pkl',
    'Naive Bayes': 'modelo_NaiveBayes_Base.pkl',
    'KNN': 'modelo_KNN_Optimizado.pkl',
    'Árbol de Decisión': 'modelo_DecisionTree_Optimizado.pkl',
    'SVM': 'modelo_SVM_Optimizado.pkl',
    'Red Neuronal MLP': 'modelo_MLP_Optimizado.pkl'
}

for nombre, filename in modelos_raw.items():
    m = safe_load(filename)
    if m:
        modelos[nombre] = m

if modelos:
    print("✅ Modelos cargados correctamente")
else:
    st.error("❌ No se pudo cargar ningún modelo. Verifica la carpeta 'models/'.")

# Bloque 2: Cargar scaler
scaler = safe_load('scaler.pkl')
if scaler:
    print("✅ Scaler cargado correctamente")

# Bloque 3: Cargar datos de test
X_test = safe_load('X_test_scaled.pkl')
y_test = safe_load('y_test.pkl')
if X_test is not None and y_test is not None:
    mostrar_matriz = True
    print("✅ Datos de test cargados correctamente")


# Interfaz
st.title("Predicción de Ataques Cardíacos")
st.markdown("")
st.markdown("")
st.markdown("")

st.markdown("""
    Esta aplicación utiliza técnicas de **Aprendizaje Automático (Machine Learning)** para **predecir el riesgo de un ataque cardíaco** en pacientes, a partir de tres biomarcadores clínicos fundamentales:

    - **Edad:** Factor de riesgo asociado al desgaste cardiovascular.
    - **CK-MB:** Enzima liberada ante daño en el músculo cardíaco.
    - **Troponina:** Proteína clave cuya presencia elevada indica daño al corazón.

    """)

st.markdown("")
with st.expander("Mostrar mas información", expanded=False):

    st.markdown("""
   
    ### Funciones principales:

    - **Predicción** -> Realiza la predicción de reisgo de ataque cardíaco para el paciente evaluado.
    - **Pobabilidad** -> muestra la probabilidad en un gráfico tipo pastel.
    - **Matriz de confusión** -> Visualiza la para evaluar el desempeño del modelo.
    - **Curvas ROC** -> Grafica que muestra la relación entre la tasa de TP y FP de todos los modelos.
    - **Tiempo de inferencia** -> Evalua el **tiempo de inferencia** para cada modelo.
                
    ---
                
    ### Modelos implementados:
    - Regresión Logística
    - Naive Bayes
    - KNN
    - Árbol de Decisión
    - SVM
    - Red Neuronal MLP
    ---
                
    ### Metricas de evaluación:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1-score**
    - **F0.5-score**
    - **F2-score**

    ---
    """)


st.markdown("### Uso de la aplicación:")
st.markdown("")
st.markdown("")
# Sidebar para configuración global
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=100)
    st.title("Configuración")
    
    st.markdown("### Seleccione el Modelo")
    nombre_modelo_sel = st.selectbox(
        "Modelo Predictivo",
        list(modelos_raw.keys()),
        help="Cada modelo utiliza un algoritmo diferente. El Árbol de Decisión y la Red Neuronal suelen ser los más precisos en este dataset."
    )
    
    st.markdown("---")
    st.markdown("### Datos del Paciente")
    edad = st.slider("Edad", 0, 120, 45, help="La edad es un factor de riesgo acumulativo importante.")
    ckmb = st.number_input("CK-MB (U/L)", value=2.86, min_value=0.0, format="%.2f", help="Creatina Quinasa-MB. Valores elevados sugieren daño reciente al músculo cardíaco.")
    troponina = st.number_input("Troponina (ng/mL)", value=0.003, min_value=0.0, format="%.3f", step=0.001, help="Proteína altamente específica. Es el 'estándar de oro' para detectar infartos.")

# Transformar datos
ckmb_log = np.log(ckmb + 1e-10)
troponina_log = np.log(troponina + 1e-10)
entrada = np.array([[edad, ckmb_log, troponina_log]])
entrada_scaled = scaler.transform(entrada)

# Definir Pestañas Principales
tab1, tab2, tab3 = st.tabs(["🎯 Diagnóstico", "🔬 Exploración y Ciencia", "📊 Evaluación Técnica"])

with tab1:
    col_res, col_gauge = st.columns([1, 1])
    
    modelo = modelos[nombre_modelo_sel]
    pred = modelo.predict(entrada_scaled)[0]
    try:
        proba = modelo.predict_proba(entrada_scaled)[0]
        prob_percent = proba[1] * 100
    except:
        proba = None
        prob_percent = None

    with col_res:
        st.subheader(f"Resultado: {'🚩 Positivo' if pred == 1 else '✅ Negativo'}")
        if pred == 1:
            st.error("Se detecta un **alto riesgo** de evento cardiaco basado en los biomarcadores proporcionados.")
        else:
            st.success("Se detecta un **bajo riesgo** de evento cardiaco actual.")
            
        st.info(f"Modelo utilizado: **{nombre_modelo_sel}**")

    with col_gauge:
        if proba is not None:
            fig_prob = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_percent,
                title = {'text': "Probabilidad de Riesgo (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#c62828" if pred == 1 else "#00acc1"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}]
                }
            ))
            fig_prob.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_prob, use_container_width=True)

with tab2:
    st.header("🧬 Justificación Científica y Datos")
    
    st.markdown("""
    ### ¿Por qué estas variables?
    La selección de **Edad**, **CK-MB** y **Troponina** no es aleatoria. En cardiología, estas son las medidas estándar:
    - **Troponina**: Es la proteína más específica para el corazón. Su elevación es casi sinónimo de lesión miocárdica.
    - **CK-MB**: Se eleva rápidamente después de un infarto. Ayuda a confirmar el diagnóstico y a medir la extensión del daño.
    
    ### Manejo de Valores Atípicos (Outliers)
    En situaciones de emergencia, los niveles de enzimas pueden dispararse a valores miles de veces superiores a lo normal. Para que nuestros modelos no se confundan con estos "saltos", aplicamos una **Transformación Logarítmica**. Esto permite que el modelo entienda la progresión del riesgo sin ser sesgado por valores extremos.
    """)
    
    st.divider()
    st.subheader("Distribución de Biomarcadores en la Población")
    
    # Aquí podríamos cargar el dataset original si existiera, o mostrar una nota.
    # Por ahora, simularemos un gráfico comparativo del paciente actual vs valores de referencia.
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name='Paciente', x=['Edad', 'CK-MB', 'Troponina'], y=[edad/100, ckmb/10, troponina*10])) # Escalado para visualización
    st.plotly_chart(fig_comp, use_container_width=True)
    st.caption("Nota: Los valores están escalados para comparación visual. Referencia clínica: Troponina < 0.04 ng/mL es usualmente normal.")

with tab3:
    st.header("📈 Rendimiento del Sistema")
    
    # Botón para disparar análisis técnico (ya que consume recursos)
    if st.checkbox("Ejecutar Análisis Técnico Completo"):

        resumen = []
        tiempos = {}
        matrices = {}
        roc_data = {}

        for nombre, modelo in modelos.items():
            try:
                start = time.perf_counter()
                y_pred = modelo.predict(X_test)
                tiempos[nombre] = time.perf_counter() - start

                if hasattr(modelo, "predict_proba"):
                    y_score = modelo.predict_proba(X_test)[:, 1]
                elif hasattr(modelo, "decision_function"):
                    y_score = modelo.decision_function(X_test)
                else:
                    y_score = None

                if y_score is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_auc = auc(fpr, tpr)
                    roc_data[nombre] = (fpr, tpr, roc_auc)

                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                resumen.append({
                    "Modelo": nombre,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": prec,
                    "Recall": rec,
                    "F1-score": f1
                })
                matrices[nombre] = confusion_matrix(y_test, y_pred)
            except:
                continue

        opcion_vista = st.radio(
            "Seleccione Vista Técnica",
            ["Métricas", "ROC", "Matrices", "Tiempos"],
            horizontal=True
        )

        if opcion_vista == "Métricas":
            df_res = pd.DataFrame(resumen).set_index("Modelo")
            st.dataframe(df_res.style.format("{:.2%}").background_gradient(cmap='Blues'))
        
        elif opcion_vista == "ROC":
            fig_roc = go.Figure()
            for nombre, (fpr, tpr, roc_auc) in roc_data.items():
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{nombre} (AUC={roc_auc:.2f})"))
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc, use_container_width=True)
            
        elif opcion_vista == "Matrices":
            # (Mantener lógica de matrices pero simplificada)
            for nombre, cm in matrices.items():
                st.write(f"**{nombre}**")
                # Gráfico simplificado
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues')
                st.plotly_chart(fig_cm, use_container_width=True)

        elif opcion_vista == "Tiempos":
            df_t = pd.DataFrame(tiempos.items(), columns=['Modelo', 'Segundos'])
            st.plotly_chart(px.bar(df_t, x='Modelo', y='Segundos', color='Segundos'), use_container_width=True)

st.markdown("")
st.markdown("")

with st.expander("**Conclusión:**", expanded=False):
    st.markdown("""
    <div style='text-align: justify; margin-bottom: 30px;'>
    Esta aplicación es una herramienta interactiva para la <b>predicción de ataques cardíacos</b>. 
    Permite evaluar el riesgo cardiovascular de los pacientes de manera rápida y precisa, facilitando la toma de decisiones informadas en el ámbito clínico. 
    Es especialmente adecuada para entornos de apoyo clínico, educación en ciencia de datos o exploración de modelos predictivos.
    </div>
    """, unsafe_allow_html=True)

