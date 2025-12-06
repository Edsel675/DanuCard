# Evidencia 4: Tren de Pensamiento y Recomendaciones

## Dashboard de Predicción de Churn - Danu Analítica

---

## Índice

1. [Descripción del Problema y Objetivo](#1-descripción-del-problema-y-objetivo)
2. [Descripción Detallada de la Limpieza de Datos](#2-descripción-detallada-de-la-limpieza-de-datos)
3. [Resultados Más Importantes de Estadística](#3-resultados-más-importantes-de-estadística)
4. [Modelos de Machine Learning Realizados](#4-modelos-de-machine-learning-realizados)
5. [Descripción del Modelo Seleccionado](#5-descripción-del-modelo-seleccionado)
6. [Tableros Utilizados](#6-tableros-utilizados)
7. [Sugerencias y Recomendaciones Generales](#7-sugerencias-y-recomendaciones-generales)
8. [Conclusión General](#8-conclusión-general)
9. [Bibliografía](#9-bibliografía)

---

## 1. Descripción del Problema y Objetivo

### 1.1 Contexto del Problema

El **churn** (abandono de clientes) representa uno de los mayores desafíos para las empresas del sector fintech. La pérdida de clientes no solo implica una reducción directa en los ingresos, sino también costos asociados a la adquisición de nuevos clientes, que pueden ser entre 5 y 25 veces más costosos que retener a los existentes.

**Problemática específica identificada:**

- Tasa de churn mensual elevada que afecta la rentabilidad del negocio
- Dificultad para identificar anticipadamente qué clientes están en riesgo de abandonar
- Falta de herramientas predictivas para toma de decisiones proactivas
- Necesidad de optimizar los esfuerzos de retención enfocándose en clientes de alto valor en riesgo

### 1.2 Objetivo General

Desarrollar un **sistema integral de predicción y análisis de churn** que permita:

1. **Identificar proactivamente** clientes en riesgo de abandonar antes de que lo hagan
2. **Cuantificar la probabilidad** de churn para cada cliente individual
3. **Proyectar tendencias futuras** de la tasa de churn a nivel agregado
4. **Facilitar la toma de decisiones** basadas en datos para estrategias de retención
5. **Evaluar el desempeño** de los agentes de atención al cliente

### 1.3 Objetivos Específicos

| Objetivo | Métrica de Éxito |
|----------|------------------|
| Predecir churn con alta precisión | AUC-ROC > 85% |
| Detectar la mayoría de casos reales | Recall > 75% |
| Visualizar tendencias históricas | Dashboard interactivo funcional |
| Proyectar escenarios futuros | Simulador con intervalos de confianza |
| Identificar clientes de alto riesgo | Clasificación por niveles de riesgo |

---

## 2. Descripción Detallada de la Limpieza de Datos

### 2.1 Fuentes de Datos

El proyecto utiliza múltiples fuentes de datos integradas:

| Archivo | Descripción | Registros |
|---------|-------------|-----------|
| `BaseDeDatos.csv` | Historial completo de transacciones y usuarios | 872,249 |
| `resultado_churn_por_mes.csv` | Datos agregados de churn mensual | ~10.4M líneas |
| `debug_central_period_last_report_v2_filtrado.csv` | Reportes de atención al cliente | 105,098 |
| `agent_score_central_period_v2.csv` | Evaluación de desempeño de agentes | 761 |

### 2.2 Proceso de Limpieza de Datos

#### A) Tratamiento de Valores Nulos

```
Estrategia aplicada por tipo de variable:

Variables Numéricas:
├── cc_csats_mean (satisfacción del cliente)
│   ├── Creación de flag 'has_cc_contact' para indicar si hubo contacto
│   └── Imputación con 0 para usuarios sin contacto
│
├── avg_gap_days (promedio días entre transacciones)
│   └── Imputación con la mediana de la variable
│
└── Otras variables numéricas
    └── Imputación con mediana o 0 según contexto

Variables Categóricas:
├── usertype (tipo de usuario)
│   └── One-Hot Encoding con drop_first=True
│
└── Otras categóricas
    └── Codificación dummy con manejo de categorías ausentes
```

#### B) Eliminación de Data Leakage

**Problema identificado:** La variable `recency_days` (días desde última transacción) estaba directamente correlacionada con la definición de churn (churn = recency_days >= 42), lo que causaba **data leakage**.

**Solución implementada:**
- Se eliminó `recency_days` como variable predictora del modelo
- Se utilizó únicamente como variable objetivo para definir el churn real
- Se implementó validación para detectar usuarios ya churneados

#### C) Eliminación de Variables Constantes

Se identificaron y eliminaron variables que:
- Tenían varianza cero o muy baja
- No aportaban información predictiva
- Estaban altamente correlacionadas (>0.95) con otras variables

#### D) Normalización de Datos

```python
# Proceso de normalización aplicado
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
# Media = 0, Desviación estándar = 1
```

### 2.3 Creación de Variables Derivadas

| Variable Creada | Fórmula | Descripción |
|-----------------|---------|-------------|
| `tx_per_month` | tx_count / tenure_months | Transacciones promedio por mes |
| `tx_per_contact` | tx_count / cc_contacts | Transacciones por contacto de soporte |
| `has_cc_contact` | 1 si cc_csats_mean != null, 0 si no | Flag de contacto con soporte |
| `tenure_months` | tenure_days / 30 | Antigüedad en meses |

### 2.4 Filtrado de Datos

```
Criterios de filtrado aplicados:
├── Usuarios activos: recency_days < 42
├── Usuarios con transacciones: tx_count > 0
├── Datos con fechas válidas: fecha_mes no nula
└── Exclusión de outliers extremos en montos
```

---

## 3. Resultados Más Importantes de Estadística

### 3.1 Estadísticas Descriptivas Generales

| Métrica | Valor |
|---------|-------|
| Total de usuarios analizados | 872,249 |
| Reportes de atención procesados | 105,098 |
| Agentes evaluados | 761 |
| Período de datos | Múltiples meses |
| Tasa de churn promedio histórica | ~33-34% |

### 3.2 Distribución de Variables Clave

#### Distribución de Transacciones (tx_count)
- **Media**: Variable según período
- **Mediana**: Menor que la media (distribución sesgada a la derecha)
- **Observación**: Los clientes con más de 5 transacciones mensuales tienen menor probabilidad de churn

#### Distribución de Días sin Transacciones (recency_days)
- **Umbral de churn**: 42 días
- **Hallazgo**: Clientes con >60 días sin transacciones tienen >70% probabilidad de churn

### 3.3 Correlaciones Significativas

| Variables | Correlación | Interpretación |
|-----------|-------------|----------------|
| tx_count vs churn | Negativa | Más transacciones → Menor churn |
| recency_days vs churn | Positiva fuerte | Más días inactivo → Mayor churn |
| amount_sum vs churn | Negativa | Mayor monto → Menor churn |
| tenure_months vs churn | Negativa moderada | Mayor antigüedad → Menor churn |

### 3.4 Análisis de Segmentación

```
Segmentación por Riesgo (basado en modelo ML):
├── Bajo (< 30%):     ~45% de usuarios
├── Medio (30-50%):   ~25% de usuarios
├── Alto (50-70%):    ~18% de usuarios
└── Crítico (≥ 70%):  ~12% de usuarios
```

### 3.5 Hallazgos Estadísticos Clave

1. **Estacionalidad**: El churn muestra variaciones estacionales significativas
2. **Relación Ingresos-Churn**: Correlación negativa entre ingresos y tasa de churn
3. **Patrón de Inactividad**: Períodos de inactividad >30 días son indicadores críticos
4. **Segmento VIP**: Menor propensión al churn comparado con otros segmentos

---

## 4. Modelos de Machine Learning Realizados

### 4.1 Modelos Explorados

Durante el desarrollo del proyecto se exploraron múltiples algoritmos de Machine Learning:

| Modelo | AUC-ROC | Precisión | Recall | F1-Score | Observaciones |
|--------|---------|-----------|--------|----------|---------------|
| **Regresión Logística** | ~75% | ~55% | ~65% | ~60% | Baseline simple, interpretable pero limitado en relaciones no lineales |
| **Árbol de Decisión** | ~78% | ~58% | ~70% | ~63% | Tendencia a sobreajuste, interpretable |
| **Gradient Boosting** | ~86% | ~62% | ~76% | ~68% | Buen rendimiento pero más lento |
| **XGBoost** | ~87% | ~63% | ~78% | ~70% | Alto rendimiento, más complejo de tunear |
| **Random Forest** | **~88.5%** | **~64.9%** | **~79.7%** | **~71.6%** | **Mejor balance rendimiento/interpretabilidad** |

### 4.2 Análisis de Resultados por Modelo

#### Regresión Logística
- **Ventajas**: Interpretabilidad, coeficientes claros
- **Limitaciones**: No captura relaciones no lineales
- **Resultado**: Descartado por bajo rendimiento predictivo

#### Árbol de Decisión
- **Ventajas**: Muy interpretable, visualizable
- **Limitaciones**: Sobreajuste severo, alta varianza
- **Resultado**: Descartado por inestabilidad

#### Gradient Boosting / XGBoost
- **Ventajas**: Alto rendimiento, maneja bien datos desbalanceados
- **Limitaciones**: Mayor tiempo de entrenamiento, más parámetros a tunear
- **Resultado**: Buenos resultados pero más complejo de mantener

#### Random Forest (Seleccionado)
- **Ventajas**: Balance óptimo entre rendimiento e interpretabilidad
- **Limitaciones**: Menos interpretable que modelos lineales
- **Resultado**: **Mejor opción** considerando todos los factores

### 4.3 Criterios de Selección

```
Criterios evaluados para selección del modelo final:
├── Rendimiento Predictivo
│   ├── AUC-ROC: Capacidad de discriminación
│   ├── Recall: Detección de casos reales de churn
│   └── F1-Score: Balance precisión-recall
│
├── Robustez
│   ├── Estabilidad ante variaciones en datos
│   ├── Manejo de datos desbalanceados
│   └── Resistencia a outliers
│
├── Interpretabilidad
│   ├── Importancia de variables
│   └── Explicabilidad de predicciones
│
└── Implementación
    ├── Tiempo de inferencia
    ├── Facilidad de despliegue
    └── Mantenibilidad
```

---

## 5. Descripción del Modelo Seleccionado

### 5.1 Modelo: Random Forest Classifier

El modelo seleccionado es un **Random Forest (Bosque Aleatorio)** de la librería Scikit-learn, optimizado para la predicción de churn.

#### Arquitectura del Modelo

```
Random Forest Classifier
├── Algoritmo: RandomForestClassifier (sklearn)
├── Número de árboles: 250
├── Profundidad máxima: 25 niveles
├── Mínimo muestras para split: 50
├── Mínimo muestras por hoja: 25
├── Criterio: Gini
└── Pesos de clase: Balanceados (class_weight='balanced')
```

### 5.2 Métricas de Rendimiento

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **AUC-ROC** | 88.5% | Excelente capacidad de discriminación |
| **Precisión** | 64.9% | De los predichos como churn, 65% realmente abandonan |
| **Recall** | 79.7% | Detecta el 80% de los casos reales de churn |
| **F1-Score** | 71.6% | Balance óptimo entre precisión y recall |
| **Exactitud (Accuracy)** | 78.5% | Clasificación correcta general |

### 5.3 Variables Predictoras Utilizadas

El modelo utiliza **11 variables** seleccionadas mediante análisis estadístico:

**Variables Numéricas (7):**
1. `tenure_months` - Antigüedad del cliente en meses
2. `tx_count` - Cantidad total de transacciones
3. `tx_per_contact` - Transacciones por contacto de soporte
4. `amount_sum` - Monto total acumulado
5. `tx_per_month` - Transacciones promedio por mes
6. `avg_gap_days` - Promedio de días entre transacciones
7. `qualification` - Calificación del cliente

**Variables Categóricas Codificadas (4):**
8. `cc_days_since_last_no hubo contacto` - Indicador sin contacto reciente
9. `usertype_HYBRID` - Tipo de usuario híbrido
10. `cc_fcr_rate_no hubo contacto` - Tasa FCR sin contacto
11. `is_premium_True` - Indicador de cliente premium

### 5.4 Importancia de Variables

```
Ranking de Importancia de Variables:
1. tx_count (transacciones)        ████████████████████ 
2. tenure_months (antigüedad)      ███████████████░░░░░ 
3. amount_sum (monto total)        ██████████████░░░░░░ 
4. avg_gap_days (días promedio)    █████████████░░░░░░░ 
5. tx_per_month (tx por mes)       ████████████░░░░░░░░ 
```

### 5.5 Pipeline de Predicción

```
Flujo de Predicción:

┌─────────────────┐
│ Datos Entrada   │
│ (Cliente nuevo) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 1. Preparación  │
│ - One-Hot Enc.  │
│ - Imputación    │
│ - Flags         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Normalización│
│ - StandardScaler│
│ - Media=0, σ=1  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Predicción   │
│ - Random Forest │
│ - 250 árboles   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Probabilidad │
│ - 0.0 a 1.0     │
│ - Umbral: 0.5   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Clasificación│
│ - Bajo (<30%)   │
│ - Medio (30-50%)│
│ - Alto (50-70%) │
│ - Crítico (≥70%)│
└─────────────────┘
```

### 5.6 Hallazgos Clave del Modelo

1. **Días sin transacciones** es el predictor más importante de churn
2. **Clientes premium** tienen menor probabilidad de abandono
3. **Mayor volumen de transacciones** correlaciona con mayor retención
4. **La antigüedad** tiene efecto protector contra el churn
5. **Contacto con soporte** puede ser indicador de problemas (mayor riesgo)

### 5.7 Ventajas del Modelo Seleccionado

| Característica | Beneficio |
|----------------|-----------|
| Class Weight Balanceado | Maneja bien el desbalance de clases |
| Múltiples Árboles (250) | Reduce varianza y sobreajuste |
| Probabilidades Calibradas | Permite clasificación por niveles de riesgo |
| Importancia de Features | Facilita interpretación de resultados |
| Robustez ante Outliers | Más estable que modelos lineales |

---

## 6. Tableros Utilizados

### 6.1 Arquitectura General del Dashboard

El dashboard fue desarrollado en **Streamlit** con una arquitectura modular:

```
Estructura del Dashboard:
├── app.py (aplicación principal - ~5,520 líneas)
├── churn_predictor.py (módulo de ML - 167 líneas)
├── churn_model.pkl (modelo entrenado)
├── churn_scaler.pkl (normalizador)
└── churn_features.json (configuración de features)
```

### 6.2 Librerías Utilizadas

| Librería | Versión | Uso |
|----------|---------|-----|
| **Streamlit** | Latest | Framework principal del dashboard |
| **Pandas** | Latest | Manipulación y análisis de datos |
| **Plotly** | Latest | Visualizaciones interactivas |
| **NumPy** | Latest | Operaciones matemáticas |
| **Scikit-learn** | Latest | Modelo de Machine Learning |

### 6.3 Vistas del Dashboard

#### Vista 1: Panel General de Churn

**Propósito:** Visión integral del comportamiento de churn en la organización.

**Componentes:**
- **KPIs Principales:**
  - Tasa de Churn (%)
  - Ingresos Mensuales (B$)
  - Registros Usuario-Mes
  - Winrate Global (%)

- **Visualizaciones:**
  - Gráfico de línea: Evolución histórica de churn
  - Gráfico de barras: Volumen de transacciones
  - Gráfico de pastel: Distribución Activos vs Churn
  - Barras horizontales: Top 5 motivos de contacto

- **Filtros:**
  - Rango de fechas
  - Tipo de análisis (todos/churn/activos)
  - Filtro por monto

#### Vista 2: Ranking de Agentes

**Propósito:** Evaluación del desempeño de agentes de atención al cliente.

**Componentes:**
- **Top 3 Agentes:** Tarjetas destacadas con métricas principales
- **Matriz de Eficiencia:** Scatter plot (Volumen vs Winrate)
- **Tabla Top 10:** Dataframe con barra de progreso para winrate

**Métricas Evaluadas:**
- Winrate (% casos ganados)
- Total de casos atendidos
- Casos ganados vs perdidos

#### Vista 3: Simulador a Futuro

**Propósito:** Proyección del comportamiento futuro del churn usando ML.

**Componentes:**
- **KPIs de Proyección:**
  - Churn Actual
  - Proyección +1 mes
  - Proyección +3 meses

- **Parámetros Configurables:**
  - Meses a proyectar (1-12)
  - Escenario (Conservador/Moderado/Optimista)
  - Ventana histórica (6/12/24 meses o todo)
  - Peso tendencia reciente vs histórica
  - Mejora esperada con intervención (0-30%)

- **Visualizaciones:**
  - Gráfico de línea dual: Histórico + Proyección
  - Intervalos de confianza (área sombreada)
  - Comparación con/sin intervención
  - Benchmarks fintech (2-5%)

#### Vista 4: Detalle de Clientes

**Propósito:** Análisis individual de clientes en riesgo.

**Componentes:**
- **Filtros Avanzados:**
  - Nivel de riesgo (Bajo/Medio/Alto/Crítico)
  - Segmento (Básico/Premium/VIP)
  - Rango de probabilidad (0-1)
  - Días sin transacciones
  - Rango de monto total

- **Tabla de Clientes:**
  - Probabilidad de churn (barra de progreso visual)
  - Nivel de riesgo
  - Días sin transacciones
  - Monto total acumulado
  - Botón de descarga CSV

### 6.4 Insights Más Importantes por Vista

#### Panel General
1. Tendencia de churn con variaciones estacionales
2. Correlación negativa ingresos-churn
3. Clientes VIP con menor propensión al churn

#### Ranking de Agentes
1. No existe correlación directa entre volumen y eficiencia
2. Top agentes manejan volúmenes moderados con alta efectividad

#### Simulador Futuro
1. Sin intervención: aumento proyectado de 3-9% en próximos meses
2. Con intervención activa: reducción potencial del 15%

#### Detalle de Clientes
1. Clientes >60 días inactivos: >70% probabilidad de churn
2. Monto total <$1,000: mayor riesgo de abandono

### 6.5 Diseño y Experiencia de Usuario

**Características de Diseño:**
- Layout ancho para máximo aprovechamiento de espacio
- Sidebar con navegación y estadísticas generales
- Tarjetas con gradientes y animaciones CSS
- Visualizaciones interactivas (zoom, hover, filtros)
- Esquema de colores consistente:
  - Azul: Datos históricos
  - Verde: Positivo/Bajo riesgo
  - Naranja: Advertencia/Riesgo medio
  - Rojo: Negativo/Alto riesgo

---

## 7. Sugerencias y Recomendaciones Generales

### 7.1 Recomendaciones Estratégicas

#### Para Reducción de Churn

| Prioridad | Recomendación | Impacto Esperado |
|-----------|---------------|------------------|
| **Alta** | Implementar alertas automáticas para clientes con probabilidad >70% | Reducción 15-20% en churn crítico |
| **Alta** | Campañas proactivas de retención para segmento de riesgo alto | Reducción 10-15% en churn general |
| **Media** | Ofertas especiales para clientes inactivos >30 días | Reactivación de 5-10% de inactivos |
| **Media** | Programa de fidelización para clientes VIP | Mantener baja tasa de churn premium |
| **Baja** | Análisis de causas raíz por motivo de contacto | Mejora en satisfacción 10-15% |

#### Para Optimización Operativa

1. **Replicar mejores prácticas** de agentes top 10%
2. **Entrenar equipos** enfocándose en balance volumen-calidad
3. **Monitorear KPIs** en tiempo real usando el dashboard
4. **Actualizar modelo** cada 3-6 meses con nuevos datos

### 7.2 Comportamiento de los Datos - Análisis

#### Patrones Identificados

```
Patrones de Comportamiento del Churn:
├── Temporal
│   ├── Estacionalidad mensual detectada
│   ├── Picos en ciertos períodos
│   └── Tendencia general creciente
│
├── Por Segmento
│   ├── Básico: Mayor tasa de churn
│   ├── Premium: Tasa media
│   └── VIP: Menor tasa de churn
│
└── Por Actividad
    ├── >5 tx/mes: Bajo riesgo
    ├── 2-5 tx/mes: Riesgo medio
    └── <2 tx/mes: Alto riesgo
```

#### Factores de Riesgo Principales

1. **Inactividad prolongada** (>30 días sin transacciones)
2. **Bajo volumen transaccional** (<2 transacciones/mes)
3. **Monto total bajo** (<$1,000 acumulado)
4. **Baja antigüedad** (<6 meses como cliente)
5. **Múltiples contactos de soporte** con temas problemáticos

### 7.3 Plan de Acción Recomendado

#### Corto Plazo (1-3 meses)
- [ ] Implementar alertas automáticas para clientes críticos
- [ ] Crear campañas de retención para segmento alto riesgo
- [ ] Establecer KPIs de seguimiento semanal

#### Mediano Plazo (3-6 meses)
- [ ] Integrar dashboard con CRM existente
- [ ] Desarrollar programa de fidelización
- [ ] Re-entrenar modelo con datos actualizados

#### Largo Plazo (6-12 meses)
- [ ] Implementar A/B testing de estrategias de retención
- [ ] Desarrollar modelos de series temporales (ARIMA/Prophet)
- [ ] Crear sistema de scoring unificado de clientes

---

## 8. Conclusión General

### 8.1 Logros del Proyecto

El proyecto ha cumplido exitosamente con los objetivos planteados:

✅ **Desarrollo de modelo predictivo de alta precisión**
- AUC-ROC de 88.5%, superando el objetivo de 85%
- Recall de 79.7%, detectando la mayoría de casos reales

✅ **Implementación de dashboard integral**
- 4 vistas especializadas para diferentes necesidades
- Visualizaciones interactivas y filtros avanzados
- Procesamiento eficiente de más de 872,000 registros

✅ **Sistema de proyección futura**
- Simulador con múltiples escenarios configurables
- Intervalos de confianza para toma de decisiones informadas
- Capacidad de evaluar impacto de intervenciones

✅ **Identificación de clientes en riesgo**
- Clasificación automática por niveles de riesgo
- Priorización basada en probabilidad y valor del cliente
- Exportación de datos para acciones operativas

### 8.2 Impacto Esperado

| Área | Impacto Proyectado |
|------|-------------------|
| **Reducción de Churn** | 15-20% con acciones preventivas |
| **Eficiencia de Retención** | 25-30% de mejora en ROI |
| **Tiempo de Análisis** | Reducción de días a minutos |
| **Toma de Decisiones** | 100% basada en datos |

### 8.3 Valor Agregado

El sistema desarrollado proporciona:

1. **Visibilidad completa** del comportamiento de churn
2. **Predicción proactiva** vs análisis reactivo tradicional
3. **Herramienta de planificación** con escenarios futuros
4. **Base para mejora continua** con modelo actualizable

### 8.4 Reflexión Final

Este proyecto demuestra cómo la **ciencia de datos aplicada** puede transformar la gestión de la relación con clientes, pasando de un enfoque reactivo a uno **predictivo y proactivo**. La combinación de Machine Learning con visualización interactiva permite no solo entender qué está sucediendo, sino también **anticipar y actuar** antes de que los problemas se materialicen.

La solución desarrollada está lista para producción y puede generar **impacto inmediato** en la reducción de churn y optimización de recursos de retención.

---

## 9. Bibliografía

### 9.1 Librerías y Frameworks

1. **Streamlit Documentation** (2024). *Streamlit - The fastest way to build data apps*. Disponible en: https://docs.streamlit.io/

2. **Scikit-learn Documentation** (2024). *Machine Learning in Python*. Disponible en: https://scikit-learn.org/stable/documentation.html

3. **Plotly Documentation** (2024). *Plotly Python Open Source Graphing Library*. Disponible en: https://plotly.com/python/

4. **Pandas Documentation** (2024). *pandas - Python Data Analysis Library*. Disponible en: https://pandas.pydata.org/docs/

5. **NumPy Documentation** (2024). *NumPy: The fundamental package for scientific computing*. Disponible en: https://numpy.org/doc/

### 9.2 Metodología y Conceptos

6. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.

7. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

8. Provost, F., & Fawcett, T. (2013). *Data Science for Business*. O'Reilly Media.

9. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

### 9.3 Recursos de Churn Prediction

10. Verbeke, W., et al. (2012). *Building comprehensible customer churn prediction models with advanced rule induction techniques*. Expert Systems with Applications.

11. Ascarza, E. (2018). *Retention Futility: Targeting High-Risk Customers Might Be Ineffective*. Journal of Marketing Research.

### 9.4 Artículos y Blogs Técnicos

12. *Understanding AUC-ROC Curve*. Towards Data Science. Disponible en: https://towardsdatascience.com/

13. *Handling Imbalanced Datasets in Machine Learning*. Analytics Vidhya. Disponible en: https://www.analyticsvidhya.com/

14. *Feature Engineering for Machine Learning*. Machine Learning Mastery. Disponible en: https://machinelearningmastery.com/

---

## Anexos

### Anexo A: Estructura de Archivos del Proyecto

```
App_churn/
├── app/
│   ├── app.py                    # Dashboard principal (5,520 líneas)
│   ├── churn_predictor.py        # Módulo de ML (167 líneas)
│   ├── churn_model.pkl           # Modelo Random Forest entrenado
│   ├── churn_scaler.pkl          # StandardScaler para normalización
│   ├── churn_features.json       # Configuración de features
│   ├── churn_model_info.json     # Información del modelo
│   ├── BaseDeDatos.csv           # Base de datos principal
│   ├── resultado_churn_por_mes.csv # Datos de churn mensual
│   ├── agent_score_central_period_v2.csv # Datos de agentes
│   ├── debug_central_period_last_report_v2_filtrado.csv # Reportes
│   ├── EXPLICACION_MODELO.md     # Documentación técnica del modelo
│   ├── README_TABLERO.md         # Documentación del tablero
│   ├── README_PITCH.md           # Presentación ejecutiva
│   ├── RESUMEN_EJECUTIVO_MODELO.md # Resumen del modelo
│   ├── Danu.png                  # Logo
│   └── venv/                     # Entorno virtual Python
└── README.md                     # Este documento
```

### Anexo B: Métricas del Modelo

| Métrica | Definición | Valor Obtenido |
|---------|-----------|----------------|
| AUC-ROC | Área bajo la curva ROC | 88.5% |
| Precisión | VP / (VP + FP) | 64.9% |
| Recall | VP / (VP + FN) | 79.7% |
| F1-Score | 2 × (P × R) / (P + R) | 71.6% |
| Accuracy | (VP + VN) / Total | 78.5% |

### Anexo C: Niveles de Riesgo

| Nivel | Rango de Probabilidad | Acción Recomendada |
|-------|----------------------|-------------------|
| Bajo | 0% - 30% | Monitoreo estándar |
| Medio | 30% - 50% | Seguimiento mensual |
| Alto | 50% - 70% | Intervención proactiva |
| Crítico | 70% - 100% | Acción inmediata |

---

**Documento elaborado como parte de la Evidencia 4 del Reto de Ciencia de Datos**

*Última actualización: Diciembre 2024*
