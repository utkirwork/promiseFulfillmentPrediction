import pandas as pd
import xgboost as xgb
import joblib
from datetime import datetime
import psycopg2
from src.data_loader import load_unscored_promises, get_connection
from src.feature_engineering import prepare_features
from src.config import MODEL_CONFIG, DB_CONFIG

def load_model():
    model_path = f'models/{MODEL_CONFIG["model_version"]}.json'
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    encoders_path = f'models/{MODEL_CONFIG["model_version"]}_encoders.pkl'
    encoders = joblib.load(encoders_path)
    
    metadata_path = f'models/{MODEL_CONFIG["model_version"]}_metadata.pkl'
    metadata = joblib.load(metadata_path)
    
    return model, encoders, metadata

def batch_predict():
    print("Model yuklanmoqda...")
    model, encoders, metadata = load_model()
    
    print("Baholanadigan ma'lumotlar yuklanmoqda...")
    df = load_unscored_promises()
    
    if len(df) == 0:
        print("Baholanadigan yangi va'dalar yo'q.")
        return
    
    print(f"Baholanadigan va'dalar soni: {len(df)}")
    
    print("\nFeaturelar tayyorlanmoqda...")
    df_features, _ = prepare_features(df, is_training=False, encoders=encoders)
    
    feature_cols = metadata['feature_columns']
    
    for col in feature_cols:
        if col not in df_features.columns:
            df_features[col] = 0
    
    X = df_features[feature_cols]
    
    print("\nPrediction qilinmoqda...")
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred_class = (y_pred_proba >= MODEL_CONFIG['threshold']).astype(int)
    
    results_df = pd.DataFrame({
        'promise_id': df['promise_id'].values,
        'p_kept': y_pred_proba,
        'class_label': y_pred_class,
        'model_version': MODEL_CONFIG['model_version']
    })
    
    print("\nNatijalar bazaga yozilmoqda...")
    conn = get_connection()
    cursor = conn.cursor()
    
    for _, row in results_df.iterrows():
        cursor.execute("""
            INSERT INTO promise_scores (promise_id, p_kept, class_label, model_version)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (promise_id) DO NOTHING
        """, (row['promise_id'], row['p_kept'], row['class_label'], row['model_version']))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"\n{len(results_df)} ta natija bazaga yozildi.")
    print(f"\nUmumiy statistika:")
    print(f"O'rtacha ehtimol: {results_df['p_kept'].mean():.2%}")
    print(f"To'lov bo'lishi mumkin (>=60%): {(results_df['p_kept'] >= 0.6).sum()} ({results_df['p_kept'].mean():.1%})")

if __name__ == '__main__':
    batch_predict()

