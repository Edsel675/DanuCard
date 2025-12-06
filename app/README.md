# 🎯 DanuCard - Dashboard de Predicción de Churn

Dashboard interactivo para análisis y predicción de churn de clientes usando Machine Learning.

## 🚀 Deploy en Streamlit Community Cloud

### Paso 1: Subir el modelo a Google Drive

El modelo de ML (`churn_model.pkl`) es demasiado grande para GitHub (317MB). Sigue estos pasos:

1. **Sube el archivo** `churn_model.pkl` a tu Google Drive
2. **Haz click derecho** en el archivo → "Compartir"
3. **Cambia el acceso** a "Cualquier persona con el enlace puede ver"
4. **Copia el enlace**, que tendrá este formato:
   ```
   https://drive.google.com/file/d/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/view
   ```
5. **Copia el ID** (la parte `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`)

### Paso 2: Configurar Streamlit Community Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con tu cuenta de GitHub
3. Click en **"New app"**
4. Configura:
   - **Repository:** `Edsel675/DanuCard`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
5. En **"Advanced settings"** → **"Secrets"**, agrega:
   ```toml
   MODEL_GDRIVE_ID = "TU_ID_DE_GOOGLE_DRIVE_AQUÍ"
   ```
6. Click en **"Deploy!"**

### Paso 3: Esperar el despliegue

- La primera vez puede tomar 5-10 minutos
- Streamlit instalará las dependencias automáticamente
- El modelo se descargará de Google Drive al iniciar

## 📁 Estructura del Proyecto

```
app/
├── app.py                 # Aplicación principal Streamlit
├── churn_predictor.py     # Clase para predicciones de ML
├── churn_model.pkl        # Modelo Random Forest (NO en GitHub - usar Google Drive)
├── churn_scaler.pkl       # Scaler para normalización
├── churn_features.json    # Configuración de features
├── churn_model_info.json  # Métricas del modelo
├── requirements.txt       # Dependencias Python
└── .streamlit/
    └── config.toml        # Configuración de tema
```

## 🔧 Desarrollo Local

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r app/requirements.txt

# Ejecutar la aplicación
cd app
streamlit run app.py
```

## 📊 Características

- ✅ Dashboard interactivo con métricas de churn
- ✅ Predicción de churn usando Random Forest
- ✅ Visualizaciones con Plotly
- ✅ Análisis por segmentos de clientes
- ✅ Exportación de resultados

## 🛠 Tecnologías

- **Frontend:** Streamlit
- **ML:** Scikit-learn (Random Forest)
- **Visualización:** Plotly
- **Data:** Pandas, NumPy

## 📝 Notas

- Los archivos CSV grandes no están incluidos en el repositorio
- El modelo se descarga automáticamente de Google Drive en la versión cloud
- Para desarrollo local, asegúrate de tener el archivo `churn_model.pkl` en la carpeta `app/`

---
Desarrollado para Danu Analítica
