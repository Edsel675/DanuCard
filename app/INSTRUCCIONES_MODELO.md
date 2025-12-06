# Instrucciones para Integrar el Modelo de Churn

## ✅ Pasos Completados

1. **Celda agregada al notebook** (`randomforest.ipynb`):
   - Celda 32: Guarda el modelo entrenado, scaler, features e información del modelo

2. **Módulo de predicción creado** (`churn_predictor.py`):
   - Clase `ChurnPredictor` para cargar y usar el modelo
   - Métodos para predecir probabilidades y clasificar riesgo

3. **Integración en Streamlit** (`app.py`):
   - Vista "Detalle Clientes": Usa el modelo para calcular probabilidades reales
   - Vista "Simulador Futuro": Usa el modelo para proyecciones

## 📋 Pasos para Ejecutar

### Paso 1: Ejecutar el Notebook

1. Abre `randomforest.ipynb`
2. Ejecuta todas las celdas hasta la celda 32 (incluyéndola)
3. La celda 32 generará los siguientes archivos en la carpeta `app/`:
   - `churn_model.pkl` - Modelo entrenado
   - `churn_scaler.pkl` - Scaler para normalización
   - `churn_features.json` - Lista de features seleccionadas
   - `churn_model_info.json` - Información del modelo

### Paso 2: Verificar Archivos Generados

Asegúrate de que estos archivos estén en la carpeta `app/`:
```
app/
├── churn_model.pkl
├── churn_scaler.pkl
├── churn_features.json
├── churn_model_info.json
├── app.py
├── churn_predictor.py
└── BaseDeDatos.csv (necesario para el modelo)
```

### Paso 3: Ejecutar la App Streamlit

```bash
cd app
streamlit run app.py
```

## 🔍 Funcionalidades Integradas

### Vista "Detalle Clientes"
- **Antes**: Probabilidad calculada como `dias_sin_transacciones / 100`
- **Ahora**: Probabilidad calculada con el modelo Random Forest entrenado
- **Riesgo**: Clasificado según probabilidad del modelo (Bajo/Medio/Alto/Crítico)

### Vista "Simulador Futuro"
- **Antes**: Extrapolación simple basada en tendencia
- **Ahora**: Combina predicciones del modelo ML con tendencias históricas
- Muestra indicador cuando se usa el modelo ML

## ⚠️ Notas Importantes

1. **BaseDeDatos.csv**: El modelo necesita este archivo para hacer predicciones. Si no está disponible, la app usará métodos alternativos (fallback).

2. **Variables Requeridas**: El modelo espera las siguientes variables (se preparan automáticamente):
   - `tenure_months`
   - `tx_count`
   - `tx_per_contact`
   - `amount_sum`
   - `tx_per_month`
   - `avg_gap_days`
   - Variables categóricas codificadas (creationflow, gender, occupation, etc.)

3. **Manejo de Errores**: Si el modelo no está disponible o hay errores, la app automáticamente usa métodos alternativos sin interrumpir la ejecución.

## 🧪 Probar la Integración

1. Ejecuta el notebook completo
2. Verifica que los archivos `.pkl` y `.json` se generaron
3. Ejecuta `streamlit run app.py`
4. Navega a "Detalle Clientes" y verifica que las probabilidades sean diferentes a las anteriores
5. Navega a "Simulador Futuro" y verifica el mensaje de éxito del modelo

## 📊 Métricas del Modelo

El modelo entrenado tiene las siguientes métricas (aproximadas):
- **Accuracy**: ~0.78
- **Precision**: ~0.64
- **Recall**: ~0.80
- **F1-Score**: ~0.72
- **AUC-ROC**: ~0.89

Estas métricas se guardan en `churn_model_info.json` y se muestran en el dashboard cuando el modelo está activo.








