"""
Script 1: ETL - Carga, Limpieza y Creación de Variable Objetivo
Autor: Jhon Olmedo
Descripción: Carga datos desde Excel, realiza limpieza, estandarización,
             ingeniería de features temporales y construcción de variable objetivo.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Configuración
# ============================================================================
DATA_PATH = r"C:\Users\jhono\Downloads\Proyecto Mineria de Datos\mdi_detenidosaprehendidos_pm_2019_2024.xlsx"

def load_data(path):
    """Carga datos desde archivo Excel"""
    print("📂 Cargando datos...")
    df = pd.read_excel(path)
    print(f"✅ Datos cargados: {df.shape[0]:,} registros × {df.shape[1]} columnas")
    return df

def explore_data(df):
    """Exploración inicial del dataset"""
    print("\n📊 Exploración de datos:")
    print(f"  - Columnas: {df.columns.tolist()}")
    print(f"\n  - Infracciones (Top 10):")
    print(f"    {df['presunta_infraccion'].value_counts().head(10).to_dict()}")
    print(f"\n  - Tipos de arma (Top 10):")
    print(f"    {df['tipo_arma'].value_counts().head(10).to_dict()}")
    print(f"\n  - Valores faltantes (Top 15):")
    print((df == "SIN_DATO").sum().sort_values(ascending=False).head(15))

def clean_data(df):
    """Limpieza y estandarización"""
    print("\n🧹 Limpiando datos...")
    df = df.copy()
    
    # Estandarizar strings
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.upper()
    
    # Convertir edad a numérico
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    
    # Reemplazar "SIN_DATO" con NA
    cols_missing = [
        "tipo_arma", "arma", "estatus_migratorio", "movilizacion",
        "condicion", "nivel_de_instruccion", "autoidentificacion_etnica"
    ]
    for col in cols_missing:
        if col in df.columns:
            df[col] = df[col].replace("SIN_DATO", pd.NA)
    
    print("✅ Datos estandarizados")
    return df

def engineer_features(df):
    """Ingeniería de features (features temporales)"""
    print("\n⚙️  Ingeniería de features...")
    
    # Features de fecha
    df["fecha_detencion_aprehension"] = pd.to_datetime(
        df["fecha_detencion_aprehension"], errors="coerce"
    )
    df["anio"] = df["fecha_detencion_aprehension"].dt.year
    df["mes"] = df["fecha_detencion_aprehension"].dt.month
    df["dia_semana"] = df["fecha_detencion_aprehension"].dt.dayofweek
    
    # Features de hora
    df["hora_detencion_aprehension"] = pd.to_datetime(
        df["hora_detencion_aprehension"], format="%H:%M:%S", errors="coerce"
    )
    df["hora"] = df["hora_detencion_aprehension"].dt.hour
    
    # Features derivadas
    df["fin_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["es_noche"] = ((df["hora"] >= 18) | (df["hora"] < 6)).astype(int)
    
    print("✅ Features creadas (año, mes, día_semana, hora, fin_semana, es_noche)")
    return df

def create_target_variable(df):
    """Creación de variable objetivo: alta_gravedad"""
    print("\n🎯 Creando variable objetivo...")
    
    alta_lista = [
        "DELITOS CONTRA LA INVIOLABILIDAD DE LA VIDA",
        "DELITOS CONTRA LA INTEGRIDAD SEXUAL Y REPRODUCTIVA",
        "DELITOS CONTRA LA INTEGRIDAD PERSONAL",
        "TERRORISMO Y SU FINANCIACIÓN"
    ]
    
    df["alta_gravedad"] = df["presunta_infraccion"].isin(alta_lista).astype(int)
    
    # Si usó arma de fuego → alta gravedad
    df.loc[df["tipo_arma"] == "ARMAS DE FUEGO", "alta_gravedad"] = 1
    
    print(f"✅ Variable objetivo creada")
    print(f"   - Casos de alta gravedad: {df['alta_gravedad'].sum():,} ({df['alta_gravedad'].mean():.2%})")
    print(f"   - Distribución por año:")
    print(f"     {df.groupby('anio')['alta_gravedad'].agg(['count', 'sum']).to_dict()}")
    
    return df

def main():
    """Pipeline ETL completo"""
    print("=" * 70)
    print("SCRIPT 1: ETL - CARGA, LIMPIEZA Y VARIABLE OBJETIVO")
    print("=" * 70)
    
    # Ejecutar pipeline
    df = load_data(DATA_PATH)
    explore_data(df)
    df = clean_data(df)
    df = engineer_features(df)
    df = create_target_variable(df)
    
    print("\n" + "=" * 70)
    print("✅ ETL COMPLETADO")
    print("=" * 70)
    
    return df

if __name__ == "__main__":
    df = main()
    # Guardar resultado intermedio
    output_path = r"C:\Users\jhono\Downloads\Proyecto_Mineria_Datos_ETL_Modelos\datos\df_procesado.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Datos procesados guardados en: {output_path}")
