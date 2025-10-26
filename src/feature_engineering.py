import pandas as pd
from src.config import CATEGORICAL_FEATURES, FEATURES

def encode_categorical_features(df, is_training=True, encoders=None):
    if is_training:
        encoded_df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            encoders[col] = df[col].unique().tolist()
        return encoded_df, encoders
    else:
        for col in CATEGORICAL_FEATURES:
            for val in encoders[col]:
                new_col = f'{col}_{val}'
                df[new_col] = (df[col] == val).astype(int)
            df = df.drop(columns=[col])
        return df, None

def prepare_features(df, is_training=True, encoders=None):
    df = df.copy()
    
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna('unknown')
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(0)
    
    if is_training:
        df_encoded, encoders = encode_categorical_features(df, is_training=True)
    else:
        df_encoded, _ = encode_categorical_features(df, is_training=False, encoders=encoders)
    
    return df_encoded, encoders

def get_feature_columns(df):
    exclude = ['promise_id', 'ticket_id', 'client_id', 'kept_label', 
               'paid_in_4d', 'promise_date', 'promised_amount', 
               'promise_days', 'late_days', 'remaining_principal', 
               'interest_rate', 'credit_product_type', 'client_age', 
               'agent_experience_days']
    
    feature_cols = [col for col in df.columns if col not in exclude and not col.startswith('promise_')]
    return sorted(feature_cols)

