import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, 
    classification_report, confusion_matrix
)
from src.data_loader import get_connection
from src.config import MODEL_CONFIG, DB_CONFIG

def load_model():
    model_path = f'models/{MODEL_CONFIG["model_version"]}.json'
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    metadata_path = f'models/{MODEL_CONFIG["model_version"]}_metadata.pkl'
    metadata = joblib.load(metadata_path)
    
    return model, metadata

def get_scored_data():
    query = """
    SELECT 
        s.promise_id,
        s.p_kept,
        s.class_label,
        f.kept_label as actual_label,
        f.promise_date
    FROM promise_scores s
    JOIN ml_promise_features_v1 f ON s.promise_id = f.promise_id
    WHERE f.kept_label IS NOT NULL
    ORDER BY s.scored_at DESC
    """
    
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df

def evaluate_model():
    print("Model metrics hisoblanmoqda...")
    
    df = get_scored_data()
    
    if len(df) == 0:
        print("Baholanayotgan ma'lumotlar topilmadi.")
        return
    
    y_true = df['actual_label']
    y_pred_proba = df['p_kept']
    y_pred = df['class_label']
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\n{'='*50}")
    print("MODEL BAHOLASH NATIJALARI")
    print(f"{'='*50}")
    print(f"\nMa'lumotlar soni: {len(df)}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    print(f"\n{'='*50}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=['Not Kept', 'Kept']))
    
    print(f"\n{'='*50}")
    print("CONFUSION MATRIX")
    print(f"{'='*50}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"                Predicted")
    print(f"              Not Kept  Kept")
    print(f"Actual Not Kept   {cm[0][0]:6d}  {cm[0][1]:4d}")
    print(f"       Kept        {cm[1][0]:6d}  {cm[1][1]:4d}")
    
    print(f"\n{'='*50}")
    print("EHTIMOLLAR TAQSIMOTI")
    print(f"{'='*50}")
    kept_mask = y_true == 1
    not_kept_mask = y_true == 0
    
    print(f"\nTo'lov qilganlar (n={kept_mask.sum()}):")
    print(f"  O'rtacha: {y_pred_proba[kept_mask].mean():.2%}")
    print(f"  Median:   {y_pred_proba[kept_mask].median():.2%}")
    
    print(f"\nTo'lov qilmaganlar (n={not_kept_mask.sum()}):")
    print(f"  O'rtacha: {y_pred_proba[not_kept_mask].mean():.2%}")
    print(f"  Median:   {y_pred_proba[not_kept_mask].median():.2%}")

if __name__ == '__main__':
    evaluate_model()

