import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'promise_fulfillment'),
    'user': os.getenv('POSTGRES_USER', 'prom_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'prom_pass')
}

MODEL_CONFIG = {
    'model_version': 'PFM_v1',
    'feature_version': 'FS_PFM_v1',
    'label_version': 'LBL_PFM_v1',
    'threshold': 0.6,
    'temporal_split_pct': 0.8,
    'random_state': 42,
    'xgboost_params': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr'
    }
}

FEATURES = [
    'promised_amount',
    'promise_days',
    'late_days',
    'remaining_principal',
    'interest_rate',
    'credit_product_type',
    'client_age',
    'agent_experience_days'
]

CATEGORICAL_FEATURES = ['credit_product_type']

