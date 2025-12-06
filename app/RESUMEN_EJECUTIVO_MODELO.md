# Resumen Ejecutivo: Modelo de Predicción de Churn

## 🎯 ¿Qué es?

Sistema de predicción de churn que combina **Machine Learning** (nivel individual) con **proyecciones temporales** (nivel agregado) para identificar clientes en riesgo y planificar estrategias de retención.

---

## 🔧 Implementación Técnica

### Modelo Base
- **Algoritmo**: Random Forest (250 árboles de decisión)
- **Rendimiento**: AUC-ROC 88.5% (excelente)
- **Entrenado con**: ~700K registros históricos

### Pipeline
```
Datos → Preparación → Normalización → Predicción ML → Clasificación de Riesgo
```

---

## 📈 Predicciones Futuras

### Metodología Dual

1. **Nivel Individual (ML)**:
   - Predice probabilidad de churn para cada cliente activo
   - Clasifica en: Bajo / Medio / Alto / Crítico

2. **Nivel Agregado (Proyección Temporal)**:
   - Combina tendencia histórica + predicciones ML
   - Proyecta tasa de churn mensual hacia el futuro
   - Incluye intervalos de confianza (incertidumbre crece con el tiempo)

### Fórmula Base
```
Proyección[i] = Último Valor + (Tendencia × Mes × Factor Escenario)
```

---

## 🎭 Escenarios Implementados

| Escenario | Factor | Uso |
|-----------|--------|-----|
| **Conservador** | +10% | Planificación presupuestal, "peor caso" |
| **Moderado** | Baseline | Proyección más probable (default) |
| **Optimista** | -10% | Objetivo alcanzable con mejoras |

### Simulación de Intervención
- Permite ajustar **mejora esperada** (0-30%, default: 15%)
- Visualiza diferencia entre escenario con/sin acción de retención
- **Ejemplo**: Si churn proyectado es 10%, con 15% mejora → 8.5%

---

## 📊 Características del Simulador

### Parámetros Configurables
- ✅ **Meses a proyectar**: 1-12 meses
- ✅ **Ventana histórica**: 6/12/24 meses o todo
- ✅ **Peso tendencia**: Balance entre tendencia reciente vs histórica
- ✅ **Escenario**: Conservador/Moderado/Optimista
- ✅ **Mejora con intervención**: 0-30%

### Visualizaciones
- 📈 Gráfico histórico + proyección futura
- 📊 Intervalos de confianza (área sombreada)
- 🎯 Benchmarks fintech (2-5% mensual)
- 🔄 Comparación con/sin intervención
- 📉 Métricas estadísticas (media, mediana, desviación)

---

## 💡 Cómo Explicarlo en 2 Minutos

### Versión Corta (Elevator Pitch)

> "Tenemos un modelo de **Random Forest** que predice churn con **88.5% de precisión**. El sistema funciona en dos niveles: primero identifica clientes individuales en riesgo usando ML, y segundo proyecta la tasa de churn agregada hacia el futuro combinando esas predicciones con tendencias históricas. Incluimos **3 escenarios** (Conservador, Moderado, Optimista) y simulamos el impacto de estrategias de retención, permitiendo planificar con conocimiento de la incertidumbre."

### Versión Técnica

> "El modelo utiliza un **RandomForestClassifier** con 250 árboles, entrenado con ~700K registros. Para predicciones futuras, combinamos probabilidades ML agregadas con proyección temporal basada en tendencia histórica. La proyección se ajusta por escenarios (factor 0.9-1.1) e incluye intervalos de confianza que crecen con la distancia temporal. El simulador permite evaluar impacto de intervenciones mediante factor de mejora configurable (0-30%)."

---

## ✅ Puntos Clave para Destacar

1. **Doble Capa**: ML individual + Proyección agregada
2. **Flexibilidad**: Múltiples escenarios y parámetros ajustables
3. **Transparencia**: Intervalos de confianza y visualización clara
4. **Accionable**: Simulación de impacto de estrategias de retención

---

## ⚠️ Limitaciones a Mencionar

- Proyecciones >6-12 meses tienen alta incertidumbre
- Asume que patrones históricos se mantienen
- Requiere datos históricos suficientes (mínimo 6 meses)

---

## 📝 Respuestas Rápidas

**P: ¿Cómo está implementado?**  
R: Random Forest con pipeline automatizado: preparación → normalización → predicción → clasificación.

**P: ¿Cómo funcionan las predicciones futuras?**  
R: Combinan ML individual (probabilidades por cliente) con tendencia histórica agregada, ajustadas por escenarios.

**P: ¿Qué escenarios hay?**  
R: Conservador (+10%), Moderado (baseline), Optimista (-10%), más simulación de intervención (0-30% mejora).

**P: ¿Qué tan confiable es?**  
R: AUC-ROC 88.5% a nivel individual. Proyecciones agregadas incluyen intervalos de confianza que reflejan incertidumbre.

---

**Documento completo**: Ver `EXPLICACION_MODELO.md` para detalles técnicos.
