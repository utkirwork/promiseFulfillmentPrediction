import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import joblib
from datetime import datetime
import os
from src.data_loader import load_promise_data
from src.feature_engineering import prepare_features, get_feature_columns
from src.config import MODEL_CONFIG

def calculate_scale_pos_weight(y):
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    if pos_count == 0:
        return 1.0
    return neg_count / pos_count

def train_model():
    print("Ma'lumotlar yuklanmoqda...")
    df = load_promise_data()
    print(f"Yuklangan yozuvlar: {len(df)}")
    
    if len(df) < 100:
        print("Xato: Ma'lumotlar soni yetarli emas. Minimal 100 yozuv kerak.")
        return
    
    print("\nMa'lumotlar tayyorlanmoqda...")
    df_features, encoders = prepare_features(df, is_training=True)
    
    feature_cols = get_feature_columns(df_features)
    print(f"Featurelar soni: {len(feature_cols)}")
    
    X = df_features[feature_cols]
    y = df_features['kept_label']
    
    print("\nTrain/Validation split qilinmoqda...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=1 - MODEL_CONFIG['temporal_split_pct'],
        random_state=MODEL_CONFIG['random_state'],
        stratify=y
    )
    
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"Class imbalance ratio: {scale_pos_weight:.2f}")
    
    params = MODEL_CONFIG['xgboost_params'].copy()
    params['scale_pos_weight'] = scale_pos_weight
    params['random_state'] = MODEL_CONFIG['random_state']
    
    print("\nModel tayyorlanmoqda...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    print("\nModel baholanmoqda...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\nModel natijalari:")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{MODEL_CONFIG["model_version"]}.json'
    model.save_model(model_path)
    print(f"\nModel saqlandi: {model_path}")
    
    encoders_path = f'models/{MODEL_CONFIG["model_version"]}_encoders.pkl'
    joblib.dump(encoders, encoders_path)
    print(f"Encoders saqlandi: {encoders_path}")
    
    metadata = {
        'model_version': MODEL_CONFIG['model_version'],
        'training_date': datetime.now().isoformat(),
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'feature_columns': feature_cols
    }
    
    metadata_path = f'models/{MODEL_CONFIG["model_version"]}_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"Metadata saqlandi: {metadata_path}")
    
    return model, encoders, metadata

if __name__ == '__main__':
    train_model()

