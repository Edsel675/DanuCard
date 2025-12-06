"""
Script para guardar el modelo de churn entrenado
"""
import pickle
import json
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("="*80)
print("GUARDANDO MODELO PARA PRODUCCIÓN")
print("="*80)

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

print("\n1. Cargando datos...")
archivo = 'BaseDeDatos.csv'
if not os.path.exists(archivo):
    print(f"ERROR: No se encontró {archivo}")
    print("Por favor ejecuta este script desde la carpeta app/")
    sys.exit(1)

df = pd.read_csv(archivo)
print(f"   ✓ Datos cargados: {df.shape[0]:,} filas x {df.shape[1]} columnas")

print("\n2. Limpiando datos...")
df_clean = df.copy()

# Eliminar columnas no útiles
columns_to_drop = ['id_user', 'first_tx', 'last_tx']
df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')

# Manejo de cc_csats_mean
if 'cc_csats_mean' in df_clean.columns:
    df_clean['has_cc_contact'] = (~df_clean['cc_csats_mean'].isnull()).astype(int)
    df_clean['cc_csats_mean'] = df_clean['cc_csats_mean'].fillna(0)

# Imputar avg_gap_days
if 'avg_gap_days' in df_clean.columns:
    median_value = df_clean['avg_gap_days'].median()
    df_clean['avg_gap_days'] = df_clean['avg_gap_days'].fillna(median_value)

# Eliminar columnas constantes
if 'has_transactions' in df_clean.columns:
    if df_clean['has_transactions'].nunique() == 1:
        df_clean = df_clean.drop(columns=['has_transactions'])

print(f"   ✓ Datos limpios: {df_clean.shape}")

print("\n3. Preparando variables...")
numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object', 'bool']).columns.tolist()

if 'churn' in numeric_cols:
    numeric_cols.remove('churn')
if 'churn' in categorical_cols:
    categorical_cols.remove('churn')

df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

selected_features_clean = [
    'tenure_months',
    'tx_count',
    'tx_per_contact',
    'amount_sum',
    'tx_per_month',
    'avg_gap_days',
    'cc_days_since_last_no hubo contacto',
    'usertype_HYBRID',
    'qualification',
    'cc_fcr_rate_no hubo contacto',
    'is_premium_True'
]

features_final = []
for feat in selected_features_clean:
    if feat in df_encoded.columns:
        features_final.append(feat)
    else:
        print(f"   Advertencia: Feature '{feat}' no encontrada, se creará con valor 0")
        df_encoded[feat] = 0
        features_final.append(feat)
X_selected = df_encoded[features_final].copy()
y_selected = df_clean['churn'].copy()

if y_selected.dtype == 'bool':
    y_final = y_selected.astype(int)
else:
    y_final = y_selected

print(f"   ✓ Features preparadas: {len(features_final)} variables")
print(f"   ✓ Dimensiones X: {X_selected.shape}")

print("\n4. Normalizando datos...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
X_scaled = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X_selected.index)
print("   ✓ Datos normalizados")

print("\n5. Dividiendo datos (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_final,
    test_size=0.20,
    random_state=42,
    stratify=y_final
)
print(f"   ✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

print("\n6. Entrenando modelo Random Forest...")
rf_final = RandomForestClassifier(
    n_estimators=250,
    max_depth=25,
    min_samples_split=50,
    min_samples_leaf=25,
    class_weight='balanced',
    criterion='gini',
    random_state=42,
    n_jobs=-1
)

rf_final.fit(X_train, y_train)
print("   ✓ Modelo entrenado")

print("\n7. Evaluando modelo...")
y_pred = rf_final.predict(X_test)
y_pred_proba = rf_final.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   AUC-ROC:   {auc:.4f}")

print("\n8. Guardando archivos...")
model_path = os.path.join(base_dir, 'churn_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(rf_final, f)
print(f"   ✓ Modelo guardado: {model_path}")

scaler_path = os.path.join(base_dir, 'churn_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✓ Scaler guardado: {scaler_path}")

features_path = os.path.join(base_dir, 'churn_features.json')
numeric_features = [f for f in features_final if f in numeric_cols]
categorical_features = [f for f in features_final if f not in numeric_cols]

features_info = {
    'features': features_final,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'categorical_cols_base': categorical_cols
}
with open(features_path, 'w') as f:
    json.dump(features_info, f, indent=2)
print(f"   ✓ Features guardadas: {features_path}")

model_info_path = os.path.join(base_dir, 'churn_model_info.json')
model_info = {
    'model_type': 'RandomForestClassifier',
    'hyperparameters': {
        'n_estimators': 250,
        'max_depth': 25,
        'min_samples_split': 50,
        'min_samples_leaf': 25,
        'class_weight': 'balanced',
        'criterion': 'gini',
        'random_state': 42
    },
    'metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc)
    },
    'train_size': len(X_train),
    'test_size': len(X_test),
    'n_features': len(features_final)
}
with open(model_info_path, 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"   ✓ Información del modelo guardada: {model_info_path}")

print("\n" + "="*80)
print("✅ MODELO LISTO PARA PRODUCCIÓN")
print("="*80)
print("\nArchivos generados:")
print(f"  1. {model_path}")
print(f"  2. {scaler_path}")
print(f"  3. {features_path}")
print(f"  4. {model_info_path}")
print("\n¡Ahora puedes ejecutar tu app de Streamlit!")

