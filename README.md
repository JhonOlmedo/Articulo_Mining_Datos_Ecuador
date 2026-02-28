# Predicción de Delitos de Alta Gravedad en Ecuador (2019-2024)
## Mining de Datos: ETL, Reducción de Dimensionalidad y Modelado Comparativo

---

## 📋 Descripción del Proyecto

Este proyecto implementa un **pipeline completo de machine learning** para predecir delitos de alta gravedad en Ecuador utilizando datos de detenciones y aprehensiones del Ministerio del Interior (2019-2024).

**Objetivo**: Comparar 6 modelos de clasificación binaria para identificar patrones de criminalidad grave.

### Características Principales:
- ✅ **489,847 registros** procesados
- ✅ **7 features engineered** (temporales, demográficos, contextuales)
- ✅ **6 modelos entrenados** (Logística, RandomForest, GradientBoosting, XGBoost)
- ✅ **División temporal** para evitar data leakage (2019-2023 train, 2024 test)
- ✅ **AUC-ROC** como métrica principal de evaluación

---

## 📊 Resultados Principales

| Posición | Modelo | AUC |
|----------|--------|-----|
| 🥇 | **XGBoost** | **0.8442** |
| 🥈 | **GradientBoosting** | **0.8411** |
| 🥉 | **Logística Completa** | **0.8284** |
| 4️⃣ | RandomForest | 0.8085 |
| 5️⃣ | Logística (sin arma) | 0.6547 |
| 6️⃣ | Logística (mínima) | 0.5568 |

**Insight**: El tipo de arma es fundamental para la predicción. Su exclusión reduce AUC de 0.828 → 0.655.

---

## 📁 Estructura del Proyecto

```
Proyecto_Mineria_Datos_ETL_Modelos/
├── datos/
│   ├── df_procesado.csv                    # Datos después de ETL
│   └── [archivo Excel original - descargue del servidor oficial]
├── scripts/
│   ├── etl.py                             # Carga, limpieza, features
│   ├── preprocesamiento.py               # Codificación, división temporal
│   └── modelado.py                       # Entrenamiento y evaluación
├── modelos/
│   └── [modelos serializados .pkl]
├── resultados/
│   ├── comparacion_modelos.png           # Visualización comparativa
│   └── resultados.csv                    # Tabla de resultados
├── main.py                                # Script master (ejecuta pipeline completo)
├── requirements.txt                      # Dependencias Python
├── README.md                             # Este archivo
├── METADATA.md                           # Documentación de datos
└── GITHUB_SETUP.md                       # Guía para publicar en GitHub

```

---

## 🚀 Instalación y Uso

### 1. Clonar/Descargar el Repositorio
```bash
git clone https://github.com/tuusuario/Proyecto_Mining_Datos.git
cd Proyecto_Mining_Datos
```

### 2. Crear Entorno Virtual
```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar Pipeline Completo
 master** (Recomendado para producción)
```bash
python main.py
```

Este comando ejecuta automáticamente:
1. ETL (carga, limpieza, features)
2. Preprocesamiento (codificación, división temporal)
3. Modelado (entrenamiento de 6 modelos + visualizaciones)

**Opción B: Ejecutar desde Jupyter Notebook** (Recomendado para análisis exploratorio)
```bash
jupyter notebook Articulo_JhonOlmedo
jupyter notebook main_notebook.ipynb
```

---

## 📖 Documentación de Datos

Ver [METADATA.md](METADATA.md) para:
- Descripción de columnas
- Definición de variable objetivo
- Reglas de limpieza
- Especificaciones de features

---

## 🎯 Variables y Features

### Variable Objetivo
**`alta_gravedad`** (binaria):
- **1**: Delitos grave (homicidio, asalto sexual, terrorismo) O arma de fuego usada
- **0**: Otros delitos

### Features (7 total)
| Feature | Tipo | Descripción |
|---------|------|-------------|
| `edad` | Numérico | Edad del detenido/aprehendido |
| `es_noche` | Binario | 1 si 18:00-05:59, 0 si no |
| `fin_semana` | Binario | 1 si sábado/domingo, 0 si no |
| `sexo` | Categórico | HOMBRE / MUJER / OTRO |
| `nombre_provincia` | Categórico | 24 provincias de Ecuador |
| `tipo_lugar` | Categórico | VÍA PÚBLICA, VIVIENDA, etc. |
| `tipo_arma` | Categórico | ARMAS DE FUEGO, BLANCA, NINGUNA, etc. |

---

## 🔧 Secciones del Pipeline

### 📄 Script Master (`main.py`)
Ejecuta el pipeline completo en 3 pasos secuenciales:

### 1️⃣ ETL (`scripts/etl.py`)
```python
✓ Carga datos desde Excel (489K registros)
✓ Estandarización de strings (UPPER, strip)
✓ Conversión de tipos (edad a numérico)
✓ Tratamiento de "SIN_DATO" → NA
✓ Ingeniería de features temporales (año, mes, hora, fin_semana, es_noche)
✓ Creación de variable objetivo (alta_gravedad)
```

### 2️⃣ Preprocesamiento (`scripts/preprocesamiento.py`)
```python
✓ Definición de 3 conjuntos de features (completo, sin arma, mínimo)
✓ Construcción de pipelines (Imputer + StandardScaler + OneHotEncoder)
✓ División temporal (2019-2023 train, 2024 test)
✓ Manejo de valores ausentes
✓ Normalización de features numéricos
✓ One-Hot Encoding de categóricos
```

### 3️⃣ Modelado (`scripts/modelado.py`)
```python
✓ Entrenamiento de 6 modelos:
  - Logística Completa (7 features)
  - Logística sin Arma (6 features)
  - Logística Mínima (3 features)
  - Random Forest (200 árboles)
  - Gradient Boosting (150 árboles)
  - XGBoost (200 árboles)
✓ Evaluación por AUC-ROC
✓ Matriz de confusión
✓ Feature importance
✓ Generación de visualizaciones
```

---

## 📊 Visualizaciones Generadas

1. **Comparación de AUC** - Gráfico de barras horizontal
2. **Curvas ROC** - Logística vs Random Forest vs XGBoost
3. **Feature Importance** - Top 10 variables más importantes (XGBoost)
4. **Matriz de Confusión** - Predicciones del mejor modelo

---

## 🧪 Validación Temporal

⚠️ **Importante**: El dataset usa división temporal para evitar data leakage:
- **Training**: 431,558 casos (2019-2023)
- **Testing**: 58,289 casos (2024)

Esto simula un escenario real donde entrenamos con datos históricos y evaluamos en año nuevo.

---

## 📈 Métricas de Evaluación

- **AUC-ROC**: Métrica principal (rango 0-1, 1 = perfecto)
- **Accuracy**: Exactitud general
- **Precision/Recall**: Compromiso sensibilidad-especificidad
- **F1-Score**: Media armónica
- **Matriz de Confusión**: Evaluación por clase

Todos los resultados se guardan automáticamente en:
- `resultados/resultados.csv` - Tabla comparativa de modelos
- `resultados/comparacion_modelos.png` - Visualizaciones completas

---

## 💾 Guardar Modelos

Los modelos entrenados se pueden serializar:

```python
import joblib

# Después de ejecutar main.py, los modelos están en memoria
# Para guardarlos manualmente:
joblib.dump(xgb_pipeline, 'modelos/xgb_model.pkl')
joblib.dump(rf_pipeline, 'modelos/rf_model.pkl')
joblib.dump(gb_pipeline, 'modelos/gb_model.pkl')

# Cargar modelo guardado
model = joblib.load('modelos/xgb_model.pkl')
predictions = model.predict_proba(X_new)
```

**Nota**: `main.py` no guarda automáticamente los modelos para ahorrar espacio. 
Descomenta las líneas de serialización si deseas guardarlos.

---

## 🔐 Requisitos Mínimos

- Python 3.8+
- 2GB RAM mínimo
- 5GB espacio en disco (datos + modelos)

---

## 📝 Licencia y Atribución

**Licencia**: MIT License  
**Autor**: Jhon Olmedo  
**Datos**: Ministerio del Interior - Ecuador  
**Basado en**: Políticas de reproducibilidad FAIR de MDPI

---

## 📧 Contacto y Soporte

Para preguntas, issues o sugerencias:
- 📨 Email: [tu email]
- 🐙 GitHub Issues: [enlace]
- 🔗 Dataset original: [enlace a Zenodo/Figshare]

---

## 🙏 Agradecimientos

- Ministerio del Interior Ecuador (datos)
- Comunidad scikit-learn, XGBoost
- MDPI por estándares de reproducibilidad

---

**Última actualización**: 28 de febrero de 2026  
**Versión**: 1.0

---

## ⚡ Notas Importantes

### Estructura del Proyecto

El proyecto incluye:
1. **`main.py`** - Script principal que ejecuta todo el pipeline automáticamente
2. **`scripts/`** - Módulos Python individuales (etl, preprocesamiento, modelado)
3. **Notebook original** - `Articulo_JhonOlmedo.ipynb` disponible para análisis interactivo

### Opciones de Ejecución

**Para producción/reproducibilidad**:
```bash
python main.py  # Ejecuta pipeline completo (10-15 min)
```

**Para análisis exploratorio**:
```bash
jupyter notebook Articulo_JhonOlmedo.ipynb  # Análisis interactivo con celdas
```

### Archivos Generados

Después de ejecutar `main.py`:
- ✅ `datos/df_procesado.csv` - Dataset procesado
- ✅ `resultados/resultados.csv` - Tabla AUC de modelos
- ✅ `resultados/comparacion_modelos.png` - Visualizaciones completas

### Configuración de Rutas

Si cambias la ubicación del archivo Excel, actualiza la ruta en `main.py` línea 26:
```python
DATA_PATH = r"TU\NUEVA\RUTA\data.xlsx"
```
