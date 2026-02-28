"""
SCRIPT MASTER - Ejecución completa del pipeline
Autor: Jhon Olmedo
Descripción: Ejecuta los 3 scripts en secuencia: ETL → Preprocesamiento → Modelado
"""

import sys
import os

# Agregar scripts a path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

print("=" * 80)
print("PIPELINE COMPLETO: ETL → PREPROCESAMIENTO → MODELADO")
print("=" * 80)

# ============================================================================
# PASO 1: ETL
# ============================================================================
print("\n" + "▶" * 40)
print("PASO 1: ETL - Carga, Limpieza y Features")
print("▶" * 40 + "\n")

from scripts.etl import load_data, explore_data, clean_data, engineer_features, create_target_variable

DATA_PATH = r"C:\Users\jhono\Downloads\Proyecto Mineria de Datos\mdi_detenidosaprehendidos_pm_2019_2024.xlsx"

# Ejecutar ETL
df = load_data(DATA_PATH)
explore_data(df)
df = clean_data(df)
df = engineer_features(df)
df = create_target_variable(df)

print("\n✅ ETL completado\n")

# ============================================================================
# PASO 2: Preprocesamiento
# ============================================================================
print("\n" + "▶" * 40)
print("PASO 2: Preprocesamiento - Codificación y División Temporal")
print("▶" * 40 + "\n")

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Definir features
features_completo = ["edad", "es_noche", "fin_semana", "sexo", "nombre_provincia", "tipo_lugar", "tipo_arma"]
features_sin_arma = ["edad", "es_noche", "fin_semana", "sexo", "nombre_provincia", "tipo_lugar"]
features_min = ["edad", "es_noche", "fin_semana"]

print("📋 Features definidos")

# División temporal
print("⏱️  División temporal (2019-2023 train, 2024 test)...")
df_train = df[df["anio"] < 2024].copy()
df_test = df[df["anio"] == 2024].copy()
print(f"  ✓ Train: {df_train.shape[0]:,} registros")
print(f"  ✓ Test:  {df_test.shape[0]:,} registros")

# Función auxiliar para crear preprocessor
def build_preprocessor(num_features, cat_features):
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    transformers = [("num", num_transformer, num_features)]
    if cat_features:
        transformers.append(("cat", cat_transformer, cat_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

# Procesadores para cada variante
preprocessor = build_preprocessor(["edad", "es_noche", "fin_semana"], ["sexo", "nombre_provincia", "tipo_lugar", "tipo_arma"])
preprocessor2 = build_preprocessor(["edad", "es_noche", "fin_semana"], ["sexo", "nombre_provincia", "tipo_lugar"])
preprocessor3 = build_preprocessor(["edad", "es_noche", "fin_semana"], [])

# Preparar features para cada variante
print("📊 Preparando features...")

X_train = df_train[features_completo].copy()
y_train = df_train["alta_gravedad"].copy()
X_test = df_test[features_completo].copy()
y_test = df_test["alta_gravedad"].copy()
X_train = X_train.replace({pd.NA: np.nan})
X_test = X_test.replace({pd.NA: np.nan})

X_train2 = df_train[features_sin_arma].copy()
y_train2 = df_train["alta_gravedad"].copy()
X_test2 = df_test[features_sin_arma].copy()
y_test2 = df_test["alta_gravedad"].copy()
X_train2 = X_train2.replace({pd.NA: np.nan})
X_test2 = X_test2.replace({pd.NA: np.nan})

X_train3 = df_train[features_min].copy()
y_train3 = df_train["alta_gravedad"].copy()
X_test3 = df_test[features_min].copy()
y_test3 = df_test["alta_gravedad"].copy()
X_train3 = X_train3.replace({pd.NA: np.nan})
X_test3 = X_test3.replace({pd.NA: np.nan})

print("✅ Preprocesamiento completado\n")

# ============================================================================
# PASO 3: Modelado
# ============================================================================
print("\n" + "▶" * 40)
print("PASO 3: Modelado - Entrenamiento y Evaluación")
print("▶" * 40 + "\n")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("🚀 Entrenando 6 modelos...\n")

models = {}
results = []

# 1. Logística Completa
print("1️⃣  Logística Completa (7 features)")
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
models["logistica_completa"] = (pipeline, y_pred_proba, y_test)
results.append(("Logística completa", auc))
print(f"   ✓ AUC: {auc:.4f}\n")

# 2. Logística sin Arma
print("2️⃣  Logística sin Arma (6 features)")
pipeline2 = Pipeline(steps=[
    ("preprocessor", preprocessor2),
    ("classifier", LogisticRegression(max_iter=1000))
])
pipeline2.fit(X_train2, y_train2)
y_pred_proba2 = pipeline2.predict_proba(X_test2)[:, 1]
auc2 = roc_auc_score(y_test2, y_pred_proba2)
models["logistica_sin_arma"] = (pipeline2, y_pred_proba2, y_test2)
results.append(("Logística sin arma", auc2))
print(f"   ✓ AUC: {auc2:.4f}\n")

# 3. Logística Mínima
print("3️⃣  Logística Mínima (3 features)")
pipeline3 = Pipeline(steps=[
    ("preprocessor", preprocessor3),
    ("classifier", LogisticRegression(max_iter=1000))
])
pipeline3.fit(X_train3, y_train3)
y_pred_proba3 = pipeline3.predict_proba(X_test3)[:, 1]
auc3 = roc_auc_score(y_test3, y_pred_proba3)
models["logistica_minima"] = (pipeline3, y_pred_proba3, y_test3)
results.append(("Logística mínima", auc3))
print(f"   ✓ AUC: {auc3:.4f}\n")

# 4. Random Forest
print("4️⃣  Random Forest (200 árboles)")
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1))
])
rf_pipeline.fit(X_train, y_train)
y_pred_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
models["random_forest"] = (rf_pipeline, y_pred_proba_rf, y_test)
results.append(("RandomForest", auc_rf))
print(f"   ✓ AUC: {auc_rf:.4f}\n")

# 5. Gradient Boosting
print("5️⃣  Gradient Boosting (150 árboles)")
gb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42))
])
gb_pipeline.fit(X_train, y_train)
y_pred_proba_gb = gb_pipeline.predict_proba(X_test)[:, 1]
auc_gb = roc_auc_score(y_test, y_pred_proba_gb)
models["gradient_boosting"] = (gb_pipeline, y_pred_proba_gb, y_test)
results.append(("GradientBoosting", auc_gb))
print(f"   ✓ AUC: {auc_gb:.4f}\n")

# 6. XGBoost
print("6️⃣  XGBoost (200 árboles)")
xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, eval_metric="logloss"
    ))
])
xgb_pipeline.fit(X_train, y_train)
y_pred_proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
models["xgboost"] = (xgb_pipeline, y_pred_proba_xgb, y_test)
results.append(("XGBoost", auc_xgb))
print(f"   ✓ AUC: {auc_xgb:.4f}\n")

# Tabla comparativa
print("=" * 80)
print("📊 RESULTADOS FINALES")
print("=" * 80)
df_results = pd.DataFrame(results, columns=["Modelo", "AUC"])
df_results = df_results.sort_values("AUC", ascending=False).reset_index(drop=True)
print("\n" + df_results.to_string(index=False))

# Guardar resultados
output_path = r"C:\Users\jhono\Downloads\Proyecto_Mineria_Datos_ETL_Modelos\resultados\resultados.csv"
df_results.to_csv(output_path, index=False)
print(f"\n✅ Resultados guardados en: {output_path}")

# Generar visualización comparativa
print("\n📈 Generando visualizaciones...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# AUC Bar Plot
ax1 = axes[0, 0]
colors = ['#2ecc71' if i == 0 else '#e74c3c' if i == len(df_results)-1 else '#3498db' 
          for i in range(len(df_results))]
ax1.barh(df_results["Modelo"], df_results["AUC"], color=colors)
ax1.set_xlabel("AUC Score", fontsize=11, fontweight='bold')
ax1.set_title("Comparación de Modelos por AUC", fontsize=12, fontweight='bold')
ax1.set_xlim(0.5, 0.9)
for i, (modelo, auc) in enumerate(zip(df_results["Modelo"], df_results["AUC"])):
    ax1.text(auc + 0.005, i, f'{auc:.4f}', va='center', fontweight='bold')

# Curvas ROC (Logística vs XGBoost vs RF)
ax2 = axes[0, 1]
logistica, y_prob_log, y_test_log = models["logistica_completa"]
xgb, y_prob_xgb, y_test_xgb = models["xgboost"]
rf, y_prob_rf, y_test_rf = models["random_forest"]

for nombre, y_prob, y_test_data in [
    ("Logística", y_prob_log, y_test_log),
    ("Random Forest", y_prob_rf, y_test_rf),
    ("XGBoost", y_prob_xgb, y_test_xgb)
]:
    fpr, tpr, _ = roc_curve(y_test_data, y_prob)
    auc = roc_auc_score(y_test_data, y_prob)
    ax2.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.3f})", linewidth=2)

ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatorio')
ax2.set_xlabel("False Positive Rate", fontsize=11, fontweight='bold')
ax2.set_ylabel("True Positive Rate", fontsize=11, fontweight='bold')
ax2.set_title("Curvas ROC Comparativas", fontsize=12, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(alpha=0.3)

# Feature Importance (XGBoost)
ax3 = axes[1, 0]
xgb_model = models["xgboost"][0]
feature_names = xgb_model.named_steps["preprocessor"].get_feature_names_out()
importances = xgb_model.named_steps["classifier"].feature_importances_
top_features = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=True).tail(10)

ax3.barh(top_features["feature"], top_features["importance"], color='#9b59b6')
ax3.set_xlabel("Importance", fontsize=11, fontweight='bold')
ax3.set_title("Top 10 Features - XGBoost", fontsize=12, fontweight='bold')

# Matriz de Confusión (XGBoost)
ax4 = axes[1, 1]
y_pred_xgb = xgb_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=False,
            xticklabels=['Bajo', 'Alto'], yticklabels=['Bajo', 'Alto'])
ax4.set_ylabel("Verdadero", fontsize=11, fontweight='bold')
ax4.set_xlabel("Predicho", fontsize=11, fontweight='bold')
ax4.set_title("Matriz de Confusión - XGBoost", fontsize=12, fontweight='bold')

plt.tight_layout()
output_image = r"C:\Users\jhono\Downloads\Proyecto_Mineria_Datos_ETL_Modelos\resultados\comparacion_modelos.png"
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: {output_image}")
plt.show()

print("\n" + "=" * 80)
print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
print("=" * 80)
print(f"\n📁 Resultados guardados en:")
print(f"   - {output_path}")
print(f"   - {output_image}")
print("\n🎉 ¡Listo para compartir con MDPI!")
