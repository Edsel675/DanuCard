# Explicación del Modelo de Predicción de Churn

## 📊 Resumen Ejecutivo

El sistema de predicción de churn está implementado con un **modelo de Machine Learning Random Forest** que combina análisis predictivo a nivel individual de clientes con proyecciones temporales a nivel agregado. El sistema permite evaluar diferentes escenarios futuros y el impacto de estrategias de retención.

---

## 🎯 1. Arquitectura del Modelo

### 1.1 Modelo Base: Random Forest Classifier

**Tipo de Modelo**: Random Forest (Bosque Aleatorio)
- **Algoritmo**: RandomForestClassifier de Scikit-learn
- **Árboles de decisión**: 250 árboles
- **Profundidad máxima**: 25 niveles
- **Métricas de rendimiento**:
  - **AUC-ROC**: 88.5% (excelente capacidad de discriminación)
  - **Precisión**: 64.9%
  - **Recall**: 79.7%
  - **F1-Score**: 71.6%
  - **Exactitud**: 78.5%

### 1.2 Características del Modelo

**Ventajas del Random Forest**:
- ✅ Captura relaciones no lineales complejas entre variables
- ✅ Maneja bien datos desbalanceados (usa `class_weight='balanced'`)
- ✅ Reduce sobreajuste mediante promediado de múltiples árboles
- ✅ Proporciona probabilidades calibradas (0-1) para cada cliente

**Limitaciones**:
- ⚠️ Menos interpretable que modelos lineales (pero se compensa con análisis de importancia de features)
- ⚠️ Requiere más recursos computacionales que modelos simples

---

## 🔧 2. Implementación Técnica

### 2.1 Pipeline de Predicción

El modelo sigue este flujo:

```
Datos de Clientes → Preparación de Features → Normalización → Predicción → Clasificación de Riesgo
```

**Pasos detallados**:

1. **Preparación de Features** (`_prepare_features`):
   - Codificación de variables categóricas (one-hot encoding)
   - Manejo de valores faltantes:
     - `cc_csats_mean`: Se crea flag `has_cc_contact` y se rellena con 0
     - `avg_gap_days`: Se rellena con la mediana
   - Asegura que todas las features esperadas estén presentes

2. **Normalización** (`_normalize_features`):
   - Usa un `StandardScaler` pre-entrenado
   - Normaliza todas las features al mismo rango para el modelo

3. **Predicción** (`predict_proba`):
   - Retorna probabilidades de churn (0-1) para cada cliente
   - Umbral por defecto: 0.5 (configurable)

4. **Clasificación de Riesgo** (`get_risk_level`):
   - **Bajo**: < 30%
   - **Medio**: 30-50%
   - **Alto**: 50-70%
   - **Crítico**: ≥ 70%

### 2.2 Validación de Datos

El sistema incluye validación automática de calidad de datos:
- Detecta usuarios ya churneados (recency_days >= 42)
- Verifica features críticas faltantes
- Valida tamaño mínimo de muestra

---

## 📈 3. Predicciones Futuras (Simulador)

### 3.1 Metodología de Proyección

El simulador combina **dos enfoques complementarios**:

#### A) Predicción a Nivel Individual (ML)
- El modelo Random Forest predice probabilidades para cada cliente activo
- Se agrega a nivel mensual: `tasa_churn = (probas >= 0.5).mean() * 100`

#### B) Proyección Temporal (Tendencia Histórica)
- Calcula la tendencia histórica de la tasa de churn mensual
- Extrapola hacia el futuro usando:
  - **Tendencia reciente** (últimos 3 meses) con peso configurable
  - **Tendencia histórica** (ventana seleccionable: 6, 12, 24 meses o todo)
  - **Promedio histórico** y desviación estándar para intervalos de confianza

### 3.2 Fórmula de Proyección

```python
# Para cada mes futuro i:
predicción[i] = último_valor + (tendencia * i * factor_escenario)

# Donde:
# - tendencia = (tendencia_reciente * peso) + (tendencia_histórica * (1 - peso))
# - factor_escenario ajusta según el escenario seleccionado
```

### 3.3 Intervalos de Confianza

El sistema calcula límites superior e inferior que aumentan con la distancia temporal:

```python
incertidumbre = desviación_estándar * (1 + mes_futuro * 0.1)
límite_superior = predicción + incertidumbre
límite_inferior = predicción - incertidumbre
```

Esto refleja que **la incertidumbre crece cuanto más lejos proyectamos**.

---

## 🎭 4. Escenarios Implementados

### 4.1 Tipos de Escenarios

El simulador incluye **3 escenarios** que ajustan la proyección:

| Escenario | Factor | Descripción |
|-----------|--------|-------------|
| **Conservador** | 1.1 | Asume que el churn será 10% mayor que la tendencia |
| **Moderado** | 1.0 | Sigue exactamente la tendencia histórica (default) |
| **Optimista** | 0.9 | Asume que el churn será 10% menor que la tendencia |

**Uso práctico**:
- **Conservador**: Planificación de presupuesto, escenario "peor caso"
- **Moderado**: Proyección más probable basada en datos históricos
- **Optimista**: Objetivo alcanzable con mejoras operativas

### 4.2 Proyección con Intervención

El sistema permite simular el impacto de **estrategias de retención activa**:

```python
churn_con_intervención = churn_sin_intervención * (1 - factor_mejora)
```

**Parámetros configurables**:
- **Mejora esperada con intervención**: 0-30% (default: 15%)
- Esto simula el efecto de campañas de retención, ofertas especiales, etc.

**Visualización**:
- **Línea roja punteada**: Proyección sin acción (business as usual)
- **Línea verde punteada**: Proyección con retención activa
- **Área sombreada**: Diferencia entre ambos escenarios (impacto de la intervención)

---

## 📊 5. Componentes del Sistema

### 5.1 Módulo `ChurnPredictor` (`churn_predictor.py`)

Clase principal que encapsula:
- Carga del modelo entrenado (`churn_model.pkl`)
- Carga del scaler (`churn_scaler.pkl`)
- Carga de configuración de features (`churn_features.json`)
- Métodos de predicción y validación

### 5.2 Integración en Dashboard (`app.py`)

**Sección "Simulador Futuro"**:
- Permite configurar:
  - Meses a proyectar (1-12)
  - Escenario (Conservador/Moderado/Optimista)
  - Ventana histórica (6/12/24 meses o todo)
  - Peso de tendencia reciente vs histórica
  - Mejora esperada con intervención

**Visualizaciones**:
- Gráfico de líneas con histórico + proyección
- Intervalos de confianza (área sombreada)
- Líneas de referencia (benchmarks fintech: 2-5%)
- Comparación con/sin intervención
- Métricas estadísticas (media, mediana, desviación estándar, rango)

---

## 🔍 6. Cómo Funciona en la Práctica

### 6.1 Flujo de Uso Típico

1. **Carga de Datos**:
   - Se carga el historial mensual de churn
   - Se carga la base de datos de clientes actuales

2. **Predicción Individual**:
   - Para cada cliente activo (recency_days < 42), el modelo ML calcula probabilidad de churn
   - Se clasifica en niveles de riesgo (Bajo/Medio/Alto/Crítico)

3. **Proyección Agregada**:
   - Se calcula la tasa de churn actual usando predicciones ML
   - Se combina con tendencia histórica para proyectar meses futuros

4. **Análisis de Escenarios**:
   - Usuario selecciona escenario y parámetros
   - Sistema genera proyección ajustada con intervalos de confianza
   - Se visualiza impacto de intervenciones de retención

### 6.2 Ejemplo de Interpretación

**Escenario Moderado, 3 meses, sin intervención**:
- Mes 1: 8.5% ± 1.2%
- Mes 2: 9.1% ± 1.4%
- Mes 3: 9.7% ± 1.6%

**Con intervención (15% mejora)**:
- Mes 1: 7.2% ± 1.0%
- Mes 2: 7.7% ± 1.2%
- Mes 3: 8.2% ± 1.4%

**Interpretación**: La intervención podría reducir el churn en ~1.5 puntos porcentuales en 3 meses.

---

## 🎯 7. Puntos Clave para Explicar

### ✅ Fortalezas del Sistema

1. **Doble Capa de Predicción**:
   - ML para identificar clientes en riesgo individual
   - Proyección temporal para planificación estratégica

2. **Flexibilidad**:
   - Múltiples escenarios para diferentes necesidades de planificación
   - Parámetros ajustables según contexto del negocio

3. **Transparencia**:
   - Intervalos de confianza muestran incertidumbre
   - Visualización clara de impacto de intervenciones

4. **Robustez**:
   - Validación automática de datos
   - Fallback a métodos alternativos si el ML falla

### ⚠️ Consideraciones

1. **Limitaciones Temporales**:
   - Proyecciones más allá de 6-12 meses tienen alta incertidumbre
   - Asume que patrones históricos se mantienen

2. **Dependencia de Datos**:
   - Requiere datos históricos suficientes (mínimo 6 meses recomendado)
   - Calidad de predicción depende de calidad de datos de entrada

3. **Asunciones del Modelo**:
   - El modelo ML asume que relaciones entre variables se mantienen
   - Cambios estructurales en el negocio pueden requerir reentrenamiento

---

## 📝 8. Respuestas Rápidas a Preguntas Comunes

### ¿Cómo está implementado el modelo?
- **Random Forest** con 250 árboles, entrenado con ~700K registros
- Pipeline automatizado: preparación → normalización → predicción → clasificación
- Integrado en dashboard Streamlit con validación de datos

### ¿Cómo funcionan las predicciones futuras?
- **Combinación de ML individual + tendencia histórica agregada**
- Proyección mensual con intervalos de confianza crecientes
- Ajuste por escenarios (Conservador/Moderado/Optimista)

### ¿Qué escenarios están implementados?
- **3 escenarios**: Conservador (+10%), Moderado (baseline), Optimista (-10%)
- **Simulación de intervención**: Permite ajustar mejora esperada (0-30%)
- Visualización comparativa de escenarios con/sin acción

### ¿Qué tan confiables son las proyecciones?
- **AUC-ROC de 88.5%** indica excelente capacidad predictiva a nivel individual
- Proyecciones agregadas incluyen intervalos de confianza
- Incertidumbre aumenta con distancia temporal (realista)

### ¿Cómo se usa en la práctica?
1. Identificar clientes en riesgo (ML individual)
2. Proyectar tasa de churn agregada (simulador)
3. Evaluar impacto de estrategias de retención (escenarios)
4. Tomar decisiones basadas en datos con conocimiento de incertidumbre

---

## 🔄 9. Mantenimiento y Mejoras Futuras

### Reentrenamiento Recomendado
- **Frecuencia**: Cada 3-6 meses o cuando haya cambios estructurales
- **Trigger**: Si métricas de validación caen significativamente

### Posibles Mejoras
- Incorporar variables macroeconómicas en proyecciones
- Modelos de series temporales (ARIMA, Prophet) para proyecciones agregadas
- Análisis de sensibilidad de parámetros
- A/B testing de estrategias de retención

---

**Última actualización**: Basado en análisis del código actual del sistema.
