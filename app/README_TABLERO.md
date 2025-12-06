# Tablero de Análisis de Churn - Documentación Técnica

## 1. Descripción de los Tableros Seleccionados

### Panel General de Churn
El tablero principal proporciona una visión integral del comportamiento de churn en la organización. Incluye:

- **Tasa de Churn**: Porcentaje (%) de clientes que han abandonado el servicio
- **Ingresos Mensuales**: Monto total en miles de millones de dólares (B$)
- **Registros Usuario-Mes**: Cantidad total de transacciones registradas (unidades)
- **Winrate Global**: Porcentaje (%) promedio de éxito de los agentes

### Ranking de Agentes
Tablero especializado para evaluar el desempeño de los agentes de atención al cliente:

- **Winrate**: Porcentaje (%) de casos ganados vs total de casos
- **Casos Ganados**: Cantidad de casos exitosos (unidades)
- **Total de Casos**: Cantidad total de casos atendidos (unidades)

### Simulador a Futuro
Tablero predictivo que proyecta el comportamiento futuro del churn:

- **Tasa de Churn Proyectada**: Porcentaje (%) estimado para meses futuros
- **Ingresos Proyectados**: Monto estimado en miles de millones de dólares (B$)

### Detalle de Clientes
Tablero detallado para análisis individual de clientes en riesgo:

- **Probabilidad de Churn**: Valor entre 0 y 1 (0-100%) calculado por el modelo ML
- **Nivel de Riesgo**: Clasificación categórica (Bajo, Medio, Alto, Crítico)
- **Días sin Transacciones**: Cantidad de días (unidades) desde la última transacción
- **Monto Total**: Valor en dólares ($) acumulado por cliente

---

## 2. Explicación General de la Implementación en Streamlit

### Tecnologías y Librerías Utilizadas

El tablero fue desarrollado utilizando **Streamlit** como framework principal, complementado con las siguientes librerías:

- **Plotly**: Para la creación de gráficas interactivas (gráficos de línea, barras, pastel, scatter)
- **Pandas**: Para el manejo y procesamiento de datos desde archivos CSV
- **NumPy**: Para operaciones matemáticas y estadísticas
- **Scikit-learn**: Para el modelo de Machine Learning (Random Forest)

### Arquitectura del Tablero

El tablero sigue una arquitectura modular:

1. **Capa de Datos**: Función `load_data()` con cache implementado (`@st.cache_data`) que carga y procesa los archivos CSV
2. **Capa de Modelo ML**: Módulo `ChurnPredictor` con cache de recursos (`@st.cache_resource`) que carga el modelo entrenado
3. **Capa de Presentación**: Funciones de renderizado para cada vista del tablero
4. **Sistema de Navegación**: Sidebar con radio buttons para cambiar entre vistas

### Gráficas Utilizadas

#### Panel General:
- **Gráfico de Línea (Line Chart)**: Evolución histórica de churn usando `plotly.graph_objects.Scatter` con modo `lines+markers`
- **Gráfico de Barras (Bar Chart)**: Volumen de transacciones usando `plotly.graph_objects.Bar` con escala de colores
- **Gráfico de Pastel (Pie Chart)**: Distribución de usuarios (Activos vs Churn) usando `plotly.graph_objects.Pie` con estilo donut
- **Gráfico de Barras Horizontales**: Top 5 motivos de contacto usando `plotly.graph_objects.Bar` con orientación horizontal

#### Ranking de Agentes:
- **Gráfico de Dispersión (Scatter Plot)**: Matriz de eficiencia (Volumen vs Winrate) usando `plotly.express.scatter`

#### Simulador a Futuro:
- **Gráfico de Línea Dual**: Comparación entre datos históricos (línea sólida) y proyecciones (línea punteada) usando `plotly.graph_objects.Scatter`

---

## 3. Explicación del Layout, Gráficas, Botones y Filtros

### Layout General

El tablero utiliza un diseño de **layout ancho** (`layout="wide"`) que permite aprovechar todo el espacio horizontal disponible. La estructura se divide en:

- **Sidebar (Barra Lateral)**: 
  - Navegación entre vistas (Panel General, Ranking Agentes, Simulador Futuro, Detalle Clientes)
  - Estadísticas generales (Reportes cargados, Agentes evaluados, Meses analizados, Usuarios totales)
  - Fondo con gradiente azul oscuro para diferenciación visual

- **Área Principal**: 
  - Contenido dinámico que cambia según la vista seleccionada
  - Fondo blanco con tarjetas con sombras y bordes redondeados

### Panel General - Componentes Detallados

#### Filtros de Análisis (Nuevos)
- **Rango de Fechas**: Selector de fecha inicial y final (`st.date_input`) que permite filtrar datos históricos por período
- **Tipo de Análisis**: Selector desplegable (`st.selectbox`) con opciones:
  - Todos los usuarios
  - Solo usuarios con Churn
  - Solo usuarios activos
- **Filtro por Monto**: Checkbox opcional (`st.checkbox`) que activa un slider (`st.slider`) para filtrar por rango de monto

#### KPIs (Indicadores Clave)
Cuatro tarjetas con métricas principales:
- Diseño con gradientes y animaciones CSS
- Indicadores de tendencia (flechas ↑↓) con colores condicionales
- Valores grandes y legibles con formato apropiado

#### Gráficas Principales
1. **Evolución Histórica de Churn**:
   - Tipo: Línea con área rellena
   - Eje X: Fechas (meses)
   - Eje Y: Porcentaje de churn (%)
   - Características: Línea suavizada (spline), marcadores circulares, área sombreada

2. **Volumen de Transacciones**:
   - Tipo: Barras verticales
   - Eje X: Fechas (meses)
   - Eje Y: Cantidad de transacciones
   - Características: Escala de colores azul, barras con bordes

3. **Distribución de Usuarios**:
   - Tipo: Pastel (donut)
   - Categorías: Activos, Churn
   - Características: Agujero central, colores verde/rojo, porcentajes visibles

#### Gráficas Secundarias
- **Top 5 Motivos de Contacto**: Barras horizontales con valores numéricos
- **Tendencia de Ingresos**: Línea con área rellena, valores en miles de millones

### Ranking de Agentes - Componentes

- **Top 3 Agentes**: Tarjetas individuales con:
  - Identificador del agente
  - Winrate destacado
  - Ratio de casos ganados/total
  - Diseño compacto y visual

- **Matriz de Eficiencia**: Gráfico de dispersión interactivo que muestra:
  - Eje X: Total de casos (volumen)
  - Eje Y: Winrate (%)
  - Tamaño de burbujas: Casos ganados
  - Línea de referencia: Promedio de winrate

- **Tabla Top 10**: Dataframe de Streamlit con columnas configuradas y formato de progreso para winrate

### Simulador a Futuro - Componentes

- **KPIs de Proyección**: Tres métricas con:
  - Churn Actual (último mes disponible)
  - Proyección +1 mes
  - Proyección +3 meses
  - Indicadores de cambio (↑↓) con porcentajes

- **Gráfico de Proyección**:
  - Línea azul sólida: Datos históricos reales
  - Línea amarilla punteada: Proyecciones futuras
  - Áreas sombreadas diferenciadas por color
  - Eje X: Fechas (meses)
  - Eje Y: Tasa de Churn (%)

### Detalle de Clientes - Componentes

#### Filtros Avanzados
- **Nivel de Riesgo**: Multiselect con opciones (Bajo, Medio, Alto, Crítico)
- **Segmento**: Multiselect con opciones (Básico, Premium, VIP)
- **Probabilidad de Churn**: Slider de rango (0-1)
- **Días sin Transacciones**: Slider de rango
- **Monto Total**: Slider de rango en dólares

#### Tabla de Clientes
- Dataframe interactivo de Streamlit con:
  - Columna de Probabilidad como barra de progreso visual
  - Formato de moneda para montos
  - Ordenamiento por probabilidad descendente (mayor riesgo primero)
  - Altura fija con scroll para grandes volúmenes

#### Funcionalidad Adicional
- Botón de descarga CSV para exportar resultados filtrados

---

## 4. Insights Más Importantes Encontrados

### Insights del Panel General

1. **Tendencia de Churn**:
   - El churn muestra variaciones estacionales significativas
   - Picos identificados en ciertos períodos requieren atención inmediata
   - La tasa promedio se mantiene alrededor del 33-34%

2. **Relación Ingresos-Churn**:
   - Existe correlación negativa entre ingresos y tasa de churn
   - Períodos de alto churn coinciden con menores ingresos
   - Los clientes de mayor valor (VIP) muestran menor propensión al churn

3. **Patrones de Transacciones**:
   - El volumen de transacciones es un predictor importante de retención
   - Clientes con más de 5 transacciones mensuales tienen menor probabilidad de churn
   - Los períodos de inactividad (>30 días) son indicadores críticos

4. **Motivos de Contacto**:
   - Ciertos motivos de contacto están correlacionados con mayor churn
   - La atención proactiva puede reducir significativamente el abandono
   - Los clientes que contactan por problemas técnicos tienen mayor riesgo

### Insights del Ranking de Agentes

1. **Eficiencia vs Volumen**:
   - Los agentes con mayor volumen no necesariamente tienen mejor winrate
   - Existe un punto óptimo de balance entre cantidad y calidad
   - Los agentes con winrate >80% manejan volúmenes moderados

2. **Mejores Prácticas**:
   - Los top 3 agentes comparten características comunes en su abordaje
   - El entrenamiento enfocado puede mejorar el desempeño general

### Insights del Simulador

1. **Proyecciones Futuras**:
   - Sin intervención, el churn proyecta un aumento del 3-9% en los próximos meses
   - Las tendencias históricas sugieren necesidad de acciones preventivas
   - El modelo ML identifica patrones que la extrapolación simple no captura

### Insights del Detalle de Clientes

1. **Segmentación de Riesgo**:
   - Clientes con >60 días sin transacciones tienen >70% probabilidad de churn
   - El segmento VIP tiene menor probabilidad promedio de churn
   - Los clientes con monto total <$1,000 tienen mayor riesgo

2. **Factores Predictivos**:
   - Días sin transacciones es el factor más correlacionado con churn
   - La probabilidad del modelo ML es más precisa que métodos simples
   - Existen clientes de alto valor en riesgo que requieren atención prioritaria

---

## 5. Explicación del Uso del Modelo de ML para Predicciones

### Arquitectura del Modelo

El tablero utiliza un **modelo Random Forest** entrenado con las siguientes características:

- **Tipo de Modelo**: RandomForestClassifier de Scikit-learn
- **Hiperparámetros Optimizados**:
  - `n_estimators`: 250 árboles
  - `max_depth`: 25 niveles
  - `min_samples_split`: 50 muestras
  - `min_samples_leaf`: 25 muestras
  - `class_weight`: 'balanced' (para manejar desbalance de clases)
  - `criterion`: 'gini'

- **Métricas de Rendimiento**:
  - Accuracy: ~78.5%
  - Precision: ~64.9%
  - Recall: ~79.7%
  - F1-Score: ~71.6%
  - AUC-ROC: ~88.5%

### Variables Utilizadas por el Modelo

El modelo utiliza **11 variables predictoras** seleccionadas mediante análisis estadístico:

**Variables Numéricas**:
1. `tenure_months`: Antigüedad del cliente en meses
2. `tx_count`: Cantidad total de transacciones
3. `tx_per_contact`: Transacciones por contacto
4. `amount_sum`: Monto total acumulado
5. `tx_per_month`: Transacciones por mes
6. `avg_gap_days`: Promedio de días entre transacciones
7. `qualification`: Calificación del cliente

**Variables Categóricas Codificadas**:
8. `cc_days_since_last_no hubo contacto`: Días desde último contacto
9. `usertype_HYBRID`: Tipo de usuario (híbrido)
10. `cc_fcr_rate_no hubo contacto`: Tasa de resolución en primer contacto
11. `is_premium_True`: Indicador de cliente premium

### Proceso de Predicción

#### 1. Carga del Modelo
```python
@st.cache_resource
def get_predictor():
    return ChurnPredictor()
```
El modelo se carga una sola vez y se cachea en memoria para optimizar rendimiento.

#### 2. Preprocesamiento de Datos
Antes de hacer predicciones, los datos pasan por:

- **Limpieza**: Eliminación de valores nulos y variables constantes
- **Codificación**: Variables categóricas convertidas a One-Hot Encoding
- **Imputación**: Valores faltantes imputados (medianas para numéricas, 0 para categóricas)
- **Normalización**: Estandarización usando StandardScaler (media=0, std=1)
- **Selección de Features**: Solo se utilizan las 11 variables seleccionadas

#### 3. Predicción de Probabilidades

El modelo genera probabilidades de churn (0-1) para cada cliente:

```python
probas = predictor.predict_proba(df_clientes)
```

Estas probabilidades se interpretan como:
- **0.0 - 0.3**: Bajo riesgo (verde)
- **0.3 - 0.5**: Riesgo medio (amarillo)
- **0.5 - 0.7**: Alto riesgo (naranja)
- **0.7 - 1.0**: Riesgo crítico (rojo)

#### 4. Clasificación de Riesgo

Basado en la probabilidad, se asigna un nivel de riesgo:

```python
def get_risk_level(proba):
    if proba < 0.3: return 'Bajo'
    elif proba < 0.5: return 'Medio'
    elif proba < 0.7: return 'Alto'
    else: return 'Crítico'
```

### Implementación en el Tablero

#### Vista "Detalle Clientes"
- **Cálculo en Tiempo Real**: Para cada cliente, se calcula la probabilidad usando el modelo
- **Visualización**: Probabilidades mostradas como barras de progreso
- **Filtrado**: Los usuarios pueden filtrar por nivel de riesgo calculado por el modelo

#### Vista "Simulador a Futuro"
- **Proyección Agregada**: El modelo predice churn para todos los clientes actuales
- **Tendencia Combinada**: Se combina la predicción del modelo con tendencias históricas
- **Proyección Mensual**: Se calculan proyecciones para +1, +2 y +3 meses

### Ventajas del Modelo ML vs Métodos Simples

1. **Precisión Mejorada**: 
   - El modelo ML tiene AUC-ROC de 88.5% vs métodos simples basados solo en días sin transacciones

2. **Múltiples Variables**:
   - Considera 11 factores simultáneamente vs métodos univariados

3. **No Linealidad**:
   - Random Forest captura relaciones complejas entre variables

4. **Robustez**:
   - Menos sensible a outliers que métodos lineales

### Limitaciones y Consideraciones

1. **Data Leakage Prevenido**:
   - Se eliminó la variable `recency_days` que definía directamente el churn
   - El modelo usa solo variables predictoras legítimas

2. **Actualización del Modelo**:
   - Se recomienda re-entrenar mensualmente con nuevos datos
   - Monitoreo de drift en la distribución de variables

3. **Interpretabilidad**:
   - Random Forest es menos interpretable que modelos lineales
   - Se puede complementar con análisis de importancia de variables

### Flujo Completo de Predicción

```
Datos del Cliente (BaseDeDatos.csv)
    ↓
Preprocesamiento (limpieza, codificación, normalización)
    ↓
Modelo Random Forest (churn_model.pkl)
    ↓
Probabilidad de Churn (0-1)
    ↓
Clasificación de Riesgo (Bajo/Medio/Alto/Crítico)
    ↓
Visualización en Tablero (barras, colores, tablas)
```

---

## Conclusión

El tablero de análisis de churn implementado en Streamlit proporciona una solución integral para:

- **Monitoreo en Tiempo Real**: Visualización actualizada de métricas clave
- **Análisis Predictivo**: Uso de Machine Learning para identificar clientes en riesgo
- **Toma de Decisiones**: Filtros y visualizaciones que permiten análisis detallados
- **Proyección Futura**: Estimaciones de comportamiento futuro basadas en modelos ML

La integración del modelo Random Forest permite identificar proactivamente clientes en riesgo de churn con alta precisión (AUC-ROC: 88.5%), facilitando la implementación de estrategias de retención efectivas.

