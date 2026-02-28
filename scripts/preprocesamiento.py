"""
Script 2: Preprocesamiento - Codificación y División Temporal
Autor: Jhon Olmedo
Descripción: Define features, construye pipelines de preprocesamiento,
             realiza división temporal (train/test) evitando fuga de datos.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ============================================================================
# Carga de datos procesados (salida de 01_etl.py)
# ============================================================================
PROCESSED_DATA_PATH = r"C:\Users\jhono\Downloads\Proyecto_Mineria_Datos_ETL_Modelos\datos\df_procesado.csv"

def load_processed_data(path):
    """Carga datos ya procesados por ETL"""
    print("📂 Cargando datos procesados...")
    df = pd.read_csv(path)
    print(f"✅ Datos cargados: {df.shape[0]:,} registros")
    return df

def define_features():
    """Define conjunto de features para el modelado"""
    print("\n🎨 Definiendo features...")
    
    features = {
        "completo": [
            "edad", "es_noche", "fin_semana", "sexo", "nombre_provincia",
            "tipo_lugar", "tipo_arma"
        ],
        "sin_arma": [
            "edad", "es_noche", "fin_semana", "sexo", "nombre_provincia", "tipo_lugar"
        ],
        "minimo": ["edad", "es_noche", "fin_semana"]
    }
    
    for name, feats in features.items():
        print(f"  ✓ {name}: {feats}")
    
    return features

def build_preprocessor(num_features, cat_features):
    """Construye pipeline de preprocesamiento"""
    print("\n⚙️  Construyendo pipeline de preprocesamiento...")
    
    # Transformador numérico
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Transformador categórico
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # Preprocessor completo
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ]
    )
    
    print("✅ Pipeline construido")
    return preprocessor

def temporal_split(df):
    """División temporal para evitar data leakage (2019-2023 train, 2024 test)"""
    print("\n⏱️  Realizando división temporal...")
    
    df_train = df[df["anio"] < 2024].copy()
    df_test = df[df["anio"] == 2024].copy()
    
    print(f"  ✓ Train (2019-2023): {df_train.shape[0]:,} registros ({df_train.shape[0]/len(df):.1%})")
    print(f"  ✓ Test (2024):       {df_test.shape[0]:,} registros ({df_test.shape[0]/len(df):.1%})")
    
    return df_train, df_test

def prepare_features(df_train, df_test, features_list):
    """Prepara X y y para train/test"""
    print(f"\n📋 Preparando features para modelado...")
    
    X_train = df_train[features_list].copy()
    y_train = df_train["alta_gravedad"].copy()
    
    X_test = df_test[features_list].copy()
    y_test = df_test["alta_gravedad"].copy()
    
    # Reemplazar NA con np.nan para sklearn
    X_train = X_train.replace({pd.NA: np.nan})
    X_test = X_test.replace({pd.NA: np.nan})
    
    print(f"  ✓ X_train: {X_train.shape}")
    print(f"  ✓ y_train: {y_train.shape} (class balance: {y_train.mean():.2%})")
    print(f"  ✓ X_test: {X_test.shape}")
    print(f"  ✓ y_test: {y_test.shape} (class balance: {y_test.mean():.2%})")
    
    return X_train, X_test, y_train, y_test

def main():
    """Pipeline de preprocesamiento completo"""
    print("=" * 70)
    print("SCRIPT 2: PREPROCESAMIENTO - CODIFICACIÓN Y DIVISIÓN TEMPORAL")
    print("=" * 70)
    
    # Cargar datos
    df = load_processed_data(PROCESSED_DATA_PATH)
    
    # Definir features
    features_dict = define_features()
    
    # Construir preprocessor para features completos
    num_features = ["edad", "es_noche", "fin_semana"]
    cat_features = ["sexo", "nombre_provincia", "tipo_lugar", "tipo_arma"]
    preprocessor = build_preprocessor(num_features, cat_features)
    
    # División temporal
    df_train, df_test = temporal_split(df)
    
    # Preparar features completos
    features_completo = features_dict["completo"]
    X_train, X_test, y_train, y_test = prepare_features(
        df_train, df_test, features_completo
    )
    
    # Preparar features sin arma
    features_sin_arma = features_dict["sin_arma"]
    num_features2 = ["edad", "es_noche", "fin_semana"]
    cat_features2 = ["sexo", "nombre_provincia", "tipo_lugar"]
    preprocessor2 = build_preprocessor(num_features2, cat_features2)
    
    X_train2, X_test2, y_train2, y_test2 = prepare_features(
        df_train, df_test, features_sin_arma
    )
    
    # Preparar features mínimos
    features_min = features_dict["minimo"]
    preprocessor3 = build_preprocessor(["edad", "es_noche", "fin_semana"], [])
    
    X_train3, X_test3, y_train3, y_test3 = prepare_features(
        df_train, df_test, features_min
    )
    
    print("\n" + "=" * 70)
    print("✅ PREPROCESAMIENTO COMPLETADO")
    print("=" * 70)
    
    return {
        "preprocessor": preprocessor,
        "preprocessor2": preprocessor2,
        "preprocessor3": preprocessor3,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "X_train2": X_train2, "X_test2": X_test2, "y_train2": y_train2, "y_test2": y_test2,
        "X_train3": X_train3, "X_test3": X_test3, "y_train3": y_train3, "y_test3": y_test3,
    }

if __name__ == "__main__":
    result = main()
    print("\n💾 Datos preparados listos para modelado")
