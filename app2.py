# -*- coding: utf-8 -*-
"""
Created on Tue May 27 19:21:35 2025

@author: jahop
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns



# -----------------------------
# SIMULACIÓN DE DATOS Y MODELO IA
# -----------------------------
np.random.seed(42)
n_samples = 300
personalidades = ['Antisocial', 'Narcisista', 'Ansioso', 'Introvertido', 'Extrovertido']
traumas = ['Sí', 'No']
adicciones = ['Sí', 'No']
tipos_crimen = ['Robo', 'Homicidio', 'Fraude', 'Violencia doméstica', 'Tráfico']
zonas = ['Urbana', 'Rural']
historial_violencia = ['Sí', 'No']
edades = np.random.randint(18, 60, size=n_samples)

def asignar_perfil(row):
    if row['Personalidad'] == 'Antisocial' and row['Trauma'] == 'Sí' and row['Adicciones'] == 'Sí':
        return 'Alto Riesgo'
    elif row['Personalidad'] in ['Antisocial', 'Narcisista'] and row['Historial_Violencia'] == 'Sí':
        return 'Riesgo Medio'
    elif row['Personalidad'] in ['Ansioso', 'Introvertido'] and row['Trauma'] == 'Sí':
        return 'Observación'
    else:
        return 'Controlado'

df_ml = pd.DataFrame({
    'Edad': edades,
    'Personalidad': np.random.choice(personalidades, size=n_samples),
    'Trauma': np.random.choice(traumas, size=n_samples),
    'Adicciones': np.random.choice(adicciones, size=n_samples),
    'Tipo_Crimen': np.random.choice(tipos_crimen, size=n_samples),
    'Zona': np.random.choice(zonas, size=n_samples),
    'Historial_Violencia': np.random.choice(historial_violencia, size=n_samples),
})
df_ml['Perfil generado'] = df_ml.apply(asignar_perfil, axis=1)

# Modelo de clasificación
X = pd.get_dummies(df_ml.drop(columns='Perfil generado'))
y = df_ml['Perfil generado'].astype('category').cat.codes
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
perfil_map = dict(enumerate(df_ml['Perfil generado'].astype('category').cat.categories))

# -----------------------------
# INTERFAZ STREAMLIT
# -----------------------------
st.set_page_config(page_title="Perfil Criminológico IA", layout="centered")
st.title("SECRETARÍA DE SEGURIDAD Y PROTECCIÓN CIUDADANA:  🧠 Generador de Perfiles Criminológicos con IA")
st.markdown("""
Esta herramienta simula perfiles criminológicos con apoyo de Inteligencia Artificial, análisis psicológico y cuestionarios interactivos.
""")


# Botón para descargar manual PDF
try:
    with open("perfiles.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(
        label="📄 Descargar Manual",
        data=PDFbyte,
        file_name="perfiles.pdf",
        mime="application/pdf",
        help="Descargue el manual de usuario en formato PDF"
    )
except FileNotFoundError:
    st.error("El archivo 'perfiles.pdf' no fue encontrado en el directorio del script.")

st.markdown("---")


# Entradas del usuario
st.header("🔍 Simulación de Caso")
edad = st.slider("Edad del individuo", 18, 60, 30)
personalidad = st.selectbox("Rasgo predominante de personalidad", personalidades)
trauma = st.radio("¿Tiene antecedentes traumáticos?", traumas)
adicciones = st.radio("¿Tiene historial de adicciones?", adicciones)
tipo_crimen = st.selectbox("Tipo de crimen asociado", tipos_crimen)
zona = st.radio("Zona de residencia", zonas)
h_violencia = st.radio("¿Historial de violencia previa?", historial_violencia)

# Diagnóstico psicológico
st.subheader("🧪 Diagnóstico Psicológico:")
if personalidad == "Antisocial" and trauma == "Sí" and adicciones == "Sí":
    st.error("Riesgo extremo: conducta antisocial con factores traumáticos y adicción.")
elif personalidad in ["Antisocial", "Narcisista"] and h_violencia == "Sí":
    st.warning("Tendencias manipuladoras o agresivas con historial violento.")
elif personalidad == "Ansioso" and adicciones == "Sí":
    st.info("Posible evasión a través del consumo. Requiere evaluación clínica.")
elif personalidad == "Extrovertido" and trauma == "No" and adicciones == "No":
    st.success("Perfil adaptativo. Poca evidencia de riesgos críticos.")
else:
    st.info("Perfil no concluyente. Se sugiere entrevista clínica.")

# Cuestionario interactivo
st.header("🧾 Cuestionario Interactivo")
p1 = st.selectbox("¿Qué aspecto te interesa más analizar en un caso?", ["Motivo del delito", "Psicología del agresor", "Escena del crimen"])
p2 = st.selectbox("¿Cuál es tu enfoque principal?", ["Prevención", "Rehabilitación", "Justicia"])
p3 = st.radio("¿Utilizas herramientas estadísticas para tus análisis?", ["Sí", "No"])

st.subheader("🎯 Resultado del Cuestionario:")
if p1 == "Psicología del agresor" and p2 == "Rehabilitación":
    st.success("Tienes un perfil clínico-humanista, centrado en el cambio de conducta.")
elif p1 == "Escena del crimen" and p2 == "Justicia":
    st.info("Perfil investigativo. Tu enfoque es forense y legalista.")
elif p3 == "Sí" and p2 == "Prevención":
    st.warning("Enfoque preventivo con uso de datos. Eres un perfilador estratégico.")
elif p1 == "Motivo del delito" and p2 == "Prevención":
    st.success("Perfil criminológico integral, orientado a la causa y prevención.")
else:
    st.info("Tu enfoque es mixto. Potencial para adaptarte a distintos roles.")

# Perfil IA
st.header("🤖 Perfil Generado por IA")
input_dict = {
    'Edad': [edad],
    'Personalidad_' + personalidad: [1],
    'Trauma_' + trauma: [1],
    'Adicciones_' + adicciones: [1],
    'Tipo_Crimen_' + tipo_crimen: [1],
    'Zona_' + zona: [1],
    'Historial_Violencia_' + h_violencia: [1],
}
input_df = pd.DataFrame(input_dict)

# Asegurar columnas
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]

# Predicción IA
y_pred = model.predict(input_df)[0]
perfil_predicho = perfil_map[y_pred]

st.subheader("📌 Perfil generado por IA:")
st.markdown(f"### 🔐 {perfil_predicho}")

# Descripción del perfil
descripciones_perfil = {
    "Alto Riesgo": "Este perfil representa una alta probabilidad de reincidencia y conducta delictiva severa. Se recomienda intervención urgente, tratamiento especializado y seguimiento constante.",
    "Riesgo Medio": "Perfil con antecedentes significativos que pueden derivar en reincidencia si no se interviene adecuadamente. Se sugiere monitoreo, entrevistas clínicas y programas de reintegración.",
    "Observación": "El individuo presenta factores de riesgo moderados. Es importante hacer seguimiento para prevenir desarrollo de patrones delictivos.",
    "Controlado": "Perfil con bajo riesgo actual. No hay indicadores críticos inmediatos, aunque se recomienda mantener atención sobre cambios conductuales futuros."
}
st.markdown("#### 🧾 Interpretación del Perfil:")
st.write(descripciones_perfil.get(perfil_predicho, "Perfil no clasificado. Requiere revisión especializada."))

# Mostrar base de datos
with st.expander("📊 Ver base de datos simulada"):
    st.dataframe(df_ml.head(20), use_container_width=True)

# Visualizaciones
st.header("📈 Visualizaciones de Datos")

# Gráfico de barras
st.subheader("Distribución de Perfiles Generados")
perfil_counts = df_ml['Perfil generado'].value_counts()
fig1, ax1 = plt.subplots()
sns.barplot(x=perfil_counts.index, y=perfil_counts.values, ax=ax1, palette='Set2')
ax1.set_ylabel("Frecuencia")
ax1.set_xlabel("Perfil")
st.pyplot(fig1)

# Gráfico de caja
st.subheader("Edad según Perfil Generado")
fig2, ax2 = plt.subplots()
sns.boxplot(data=df_ml, x='Perfil generado', y='Edad', palette='Set3')
st.pyplot(fig2)



# -------- Contacto --------
st.header("📬 Contacto Profesional")
st.markdown("""
**Javier Horacio Pérez Ricárdez**  
📧 [Correo: jahoperi@gmail.com](mailto:jahoperi@gmail.com)  
📱 +52 56 1056 4095  
🔗 [LinkedIn](https://www.linkedin.com/in/javier-horacio-perez-ricardez-5b3a5777/)
""")
