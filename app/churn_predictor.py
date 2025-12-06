"""
Módulo para cargar y usar el modelo de predicción de churn
"""
import pickle
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st

# URL del modelo en Google Drive (configurar después de subir el archivo)
# Para obtener el ID: Subir archivo a Google Drive -> Click derecho -> Obtener enlace -> Copiar ID
# El enlace tiene formato: https://drive.google.com/file/d/FILE_ID/view
# Usar el FILE_ID aquí:
MODEL_GDRIVE_ID = os.environ.get('MODEL_GDRIVE_ID', None)

def download_model_from_gdrive(file_id: str, destination: str):
    """Descarga un archivo desde Google Drive"""
    import urllib.request
    
    # URL para descarga directa de Google Drive
    URL = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    
    try:
        st.info("📥 Descargando modelo de Machine Learning... (esto puede tomar unos segundos)")
        urllib.request.urlretrieve(URL, destination)
        st.success("✅ Modelo descargado exitosamente")
        return True
    except Exception as e:
        st.error(f"❌ Error descargando modelo: {e}")
        return False

class ChurnPredictor:
    """Carga y usa el modelo de predicción de churn"""
    
    def __init__(self, model_dir: Optional[str] = None):
        """Inicializa el predictor. Si model_dir es None, usa el directorio actual."""
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.features_info = None
        self.model_info = None
        
        self._load_model()
        self._load_scaler()
        self._load_features()
        self._load_model_info()
    
    def _load_model(self):
        """Carga el modelo entrenado, descargándolo si es necesario"""
        model_path = os.path.join(self.model_dir, 'churn_model.pkl')
        
        # Si el modelo no existe localmente, intentar descargarlo
        if not os.path.exists(model_path):
            if MODEL_GDRIVE_ID:
                success = download_model_from_gdrive(MODEL_GDRIVE_ID, model_path)
                if not success:
                    raise FileNotFoundError(
                        f"No se pudo descargar el modelo. "
                        f"Por favor verifica que el archivo esté compartido públicamente en Google Drive."
                    )
            else:
                raise FileNotFoundError(
                    f"No se encontró el modelo en {model_path}. "
                    f"Para la versión en la nube, configura la variable MODEL_GDRIVE_ID en Streamlit Secrets."
                )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def _load_scaler(self):
        """Carga el scaler para normalización"""
        scaler_path = os.path.join(self.model_dir, 'churn_scaler.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"No se encontró el scaler en {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
    def _load_features(self):
        """Carga la información de features"""
        features_path = os.path.join(self.model_dir, 'churn_features.json')
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"No se encontró el archivo de features en {features_path}")
        with open(features_path, 'r') as f:
            self.features_info = json.load(f)
    
    def _load_model_info(self):
        """Carga información del modelo"""
        model_info_path = os.path.join(self.model_dir, 'churn_model_info.json')
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                self.model_info = json.load(f)
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara y codifica las features del dataframe"""
        df_work = df.copy()
        categorical_cols_base = self.features_info.get('categorical_cols_base', [])
        
        if 'cc_csats_mean' in df_work.columns:
            df_work['has_cc_contact'] = (~df_work['cc_csats_mean'].isnull()).astype(int)
            df_work['cc_csats_mean'] = df_work['cc_csats_mean'].fillna(0)
        
        if 'avg_gap_days' in df_work.columns:
            median_gap = df_work['avg_gap_days'].median()
            df_work['avg_gap_days'] = df_work['avg_gap_days'].fillna(median_gap)
        
        df_encoded = pd.get_dummies(
            df_work, 
            columns=[col for col in categorical_cols_base if col in df_work.columns], 
            drop_first=True
        )
        
        expected_features = self.features_info['features']
        
        for feature in expected_features:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        X = df_encoded[expected_features].copy()
        return X
    
    def _normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normaliza las features usando el scaler"""
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        return X_scaled
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades de churn (0-1) para cada cliente"""
        X = self._prepare_features(df)
        X_scaled = self._normalize_features(X)
        probas = self.model.predict_proba(X_scaled)[:, 1]
        return probas
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predice churn binario (0 o 1) usando el threshold especificado"""
        probas = self.predict_proba(df)
        predictions = (probas >= threshold).astype(int)
        return predictions
    
    def get_risk_level(self, proba: float) -> str:
        """Clasifica el nivel de riesgo según la probabilidad"""
        if proba < 0.3:
            return 'Bajo'
        elif proba < 0.5:
            return 'Medio'
        elif proba < 0.7:
            return 'Alto'
        else:
            return 'Crítico'
    
    def predict_with_risk(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Predice churn y agrega columnas de probabilidad y riesgo"""
        df_result = df.copy()
        probas = self.predict_proba(df)
        df_result['probabilidad_churn'] = probas
        df_result['prediccion_churn'] = self.predict(df, threshold)
        df_result['riesgo'] = [self.get_risk_level(p) for p in probas]
        return df_result
    
    def get_model_info(self) -> Dict:
        """Retorna información del modelo"""
        return self.model_info if self.model_info else {}
    
    def validate_data_quality(self, df: pd.DataFrame) -> dict:
        """Valida que los datos sean apropiados para predicción"""
        issues = []
        warnings = []
        
        # Check si tiene usuarios ya churneados
        if 'recency_days' in df.columns:
            churned = (df['recency_days'] >= 42).sum()
            total = len(df)
            if churned > 0:
                pct = churned/total*100
                issues.append(f"{churned} usuarios ({pct:.1f}%) ya están churneados (recency_days>=42)")
        
        # Check features críticas faltantes
        critical_features = ['tx_count', 'recency_days', 'amount_sum', 'tenure_days']
        for feat in critical_features:
            if feat in df.columns:
                missing = df[feat].isna().sum()
                if missing > 0:
                    pct = missing/len(df)*100
                    warnings.append(f"{feat}: {missing} valores nulos ({pct:.1f}%)")
            else:
                issues.append(f"Columna crítica '{feat}' no existe en los datos")
        
        # Check si hay suficientes datos
        if len(df) < 10:
            issues.append(f"Muy pocos registros para predicción confiable: {len(df)}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_records': len(df)
        }
