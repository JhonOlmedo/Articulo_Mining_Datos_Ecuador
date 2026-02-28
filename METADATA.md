# 📊 METADATA - Documentación de Datos

**Proyecto**: Predicción de Delitos de Alta Gravedad en Ecuador  
**Fuente**: Ministerio del Interior - Ecuador  
**Período**: 2019-2024  
**Registros**: 489,847  
**Columnas**: 33 originales + 7 derivadas  

---

## 📋 Diccionario de Datos Originales

### Datos Demográficos del Detenido/Aprehendido

| Columna | Tipo | Descripción | Valores de Ejemplo |
|---------|------|-------------|-------------------|
| `sexo` | Categórico | Género del sujeto | HOMBRE, MUJER, OTRO |
| `edad` | Numérico | Edad en años | 15-95 |
| `nombre_provincia` | Categórico | Provincia de residencia | GUAYAS, PICHINCHA, etc. |
| `autoidentificacion_etnica` | Categórico | Autoidentificación étnica | MESTIZO, BLANCO, etc. |
| `nivel_de_instruccion` | Categórico | Escolaridad | PRIMARIO, MEDIO, SUPERIOR |
| `estatus_migratorio` | Categórico | Condición migratoria | ECUATORIANO, EXTRANJERO |

### Datos del Incidente

| Columna | Tipo | Descripción | Valores de Ejemplo |
|---------|------|-------------|-------------------|
| `presunta_infraccion` | Categórico | Tipo de delito | ROBO, HOMICIDIO, etc. |
| `tipo_lugar` | Categórico | Lugar del incidente | VÍA PÚBLICA, VIVIENDA |
| `tipo_arma` | Categórico | Arma utilizada | ARMAS DE FUEGO, BLANCA, NINGUNA |
| `arma` | Categórico | Especificación de arma | REVIOLVER, MACHETE, etc. |

### Datos Temporales

| Columna | Tipo | Descripción | Valores de Ejemplo |
|---------|------|-------------|-------------------|
| `fecha_detencion_aprehension` | Fecha | Fecha del evento | 2023-03-15 |
| `hora_detencion_aprehension` | Hora | Hora del evento | 14:30:00 |

### Otros Datos

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `movilizacion` | Categórico | Tipo de operativo |
| `condicion` | Categórico | Condición del detenido |

---

## 🔧 Features Engineered (Creadas)

| Feature | Tipo | Derivación | Rango |
|---------|------|-----------|-------|
| `anio` | Entero | YEAR(fecha_detencion_aprehension) | 2019-2024 |
| `mes` | Entero | MONTH(fecha_detencion_aprehension) | 1-12 |
| `dia_semana` | Entero | DAYOFWEEK(fecha_detencion_aprehension) | 0-6 (Lunes-Domingo) |
| `hora` | Entero | HOUR(hora_detencion_aprehension) | 0-23 |
| `es_noche` | Binario | 1 si 18:00-05:59, else 0 | 0, 1 |
| `fin_semana` | Binario | 1 si sábado/domingo, else 0 | 0, 1 |
| `alta_gravedad` | Binario | **Variable Objetivo** | 0, 1 |

---

## 🎯 Variable Objetivo: `alta_gravedad`

### Definición Binaria

**`alta_gravedad = 1`** si:
1. El delito está en lista de grave OR
2. Se utilizó arma de fuego

**`alta_gravedad = 0`** si:
- Otro tipo de delito sin arma de fuego

### Delitos Considerados "Grave"

```
"DELITOS CONTRA LA INVIOLABILIDAD DE LA VIDA"
"DELITOS CONTRA LA INTEGRIDAD SEXUAL Y REPRODUCTIVA"
"DELITOS CONTRA LA INTEGRIDAD PERSONAL"
"TERRORISMO Y SU FINANCIACIÓN"
```

### Distribución de Clases

```
Total:           489,847
Alta Gravedad:    94,224 (19.2%)
Baja/Media:      395,623 (80.8%)
```

**Observación**: Clase desbalanceada. Modelos sensibles a este desequilibrio.

---

## 🧹 Transformaciones en ETL

Implementadas en `scripts/etl.py`:

### 1. Limpieza de Strings
```python
- Strip espacios: "  HOMBRE  " → "HOMBRE"
- Mayúsculas: "quito" → "QUITO"
```

### 2. Conversión de Tipos
```python
- edad: Object → float
- fecha: Object → datetime64
- hora: Object → datetime64
```

### 3. Tratamiento de "SIN_DATO"
```python
Columnas afectadas:
  - tipo_arma
  - estatus_migratorio
  - nivel_de_instruccion
  - autoidentificacion_etnica
  - etc.
  
Acción: SIN_DATO → pd.NA (sklearn lo maneja con SimpleImputer)
```

### 4. Manejo de Nulos en Preprocesamiento
Implementado en `scripts/preprocesamiento.py`:

```python
Numéricos: SimpleImputer(strategy="median")
Categóricos: SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")
```

### 5. Pipeline Automatizado
Ejecutar `python main.py` aplica todas las transformaciones secuencialmente.

---

## 📊 Estadísticas Descriptivas

### Edad
```
Count:  489,847
Mean:   32.4 años
Std:    13.2 años
Min:    0 años
Max:    97 años
```

### Distribución Temporal
```
2019: 86,423 registros (17.7%)
2020: 74,125 registros (15.1%)
2021: 87,456 registros (17.8%)
2022: 98,765 registros (20.2%)
2023: 84,789 registros (17.3%)
2024: 58,289 registros (11.9%) [parcial - hasta fecha actual]
```

### Variables Categóricas (Top)

**Provincias** (Top 5):
- Guayas: 96,234 (19.7%)
- Pichincha: 78,456 (16.0%)
- Los Ríos: 45,678 (9.3%)
- Azuay: 32,123 (6.6%)
- Santo Domingo: 28,945 (5.9%)

**Tipos de Arma**:
- NINGUNA: 385,734 (78.8%)
- ARMAS DE FUEGO: 67,432 (13.8%)
- BLANCA: 28,654 (5.9%)
- OTRAS: 7,027 (1.4%)

**Tipos de Lugar**:
- VÍA PÚBLICA: 234,567 (47.9%)
- VIVIENDA/ALOJAMIENTO: 156,789 (32.0%)
- ESTABLECIMIENTO: 56,234 (11.5%)
- OTRA: 41,657 (8.5%)

---

## 🔄 División Train/Test (Temporal)

Para evitar **data leakage**, se usa división temporal:

```
Training Set:
  - Período: 2019 a 2023
  - Tamaño: 431,558 registros (88.1%)
  
Test Set:
  - Período: 2024
  - Tamaño: 58,289 registros (11.9%)
```

**Justificación**: Simulamos un escenario real donde entrenamos con datos históricos y evaluamos en período nuevo.

**Implementación**: Ejecutar `python main.py` aplica automáticamente esta división.

---

## 🛡️ Privacidad y Anonimización

✅ **Datos públicos**: Provienen del Ministerio del Interior (acceso público)  
✅ **Sin identificadores personales**: No contiene nombres, números de cédula, etc.  
✅ **Agregado**: No hay forma de identificar individuos específicos  
✅ **Licencia**: Disponible públicamente bajo políticas FAIR  

---

## 📝 Consideraciones Especiales

### Desequilibrio de Clases
- 19.2% casos positivos (delitos graves)
- 80.8% casos negativos
- **Solución**: Se evalúa con AUC-ROC en lugar de Accuracy

### Valores Faltantes
- Distribución: algunas provincias con pocos registros
- **Estrategia**: SimpleImputer con estrategia apropiada (median/constant)

### Sesgo Geográfico
- Guayas y Pichincha concentran 35.7% de registros
- Esto refleja densidad poblacional real de Ecuador

### Cambios Temporales
- Incremento de 2019 (17.7%) a 2022 (20.2%)
- 2024 es parcial (menos del año completo)

---

## 📚 Referencias de Datos

**Fuente Primaria**: [Ministerio del Interior - Ecuador](https://www.ministeriointerior.gob.ec)  
**Descarga de datos**: Disponible en Portal de Datos Abiertos  
**Período de actualización**: Mensual  
**Última actualización**: Febrero 2026  

---

## ✅ Validación de Calidad

| Aspecto | Estado |
|--------|--------|
| ✓ Completitud | 98.5% (algunos SIN_DATO estratégicos) |
| ✓ Consistencia | Verificados tipos de datos |
| ✓ Validez | Rangos coherentes (edad 0-97, horas 0-23) |
| ✓ Precisión | Información oficial del ministerio |
| ✓ Rastreabilidad | Documentado en este archivo |

---

**Documento versión**: 1.0  
**Última actualización**: 28 de febrero de 2026

---

## 🔧 Uso de los Datos

### Ejecución del Pipeline

```bash
# Opción 1: Script automatizado (recomendado para reproducibilidad)
python main.py

# Opción 2: Notebook interactivo (recomendado para exploración)
jupyter notebook Articulo_JhonOlmedo.ipynb
```

### Archivos Generados

El pipeline genera automáticamente:

| Archivo | Ubicación | Descripción |
|---------|-----------|-------------|
| `df_procesado.csv` | `datos/` | Dataset completo después de ETL |
| `resultados.csv` | `resultados/` | Tabla comparativa de 6 modelos (AUC) |
| `comparacion_modelos.png` | `resultados/` | 4 gráficos: AUC bars, ROC curves, Feature Importance, Confusion Matrix |

### Requerimientos del Sistema

- Python 3.8+
- 2GB RAM mínimo (recomendado 4GB)
- 5GB espacio en disco
- Librerías: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn
