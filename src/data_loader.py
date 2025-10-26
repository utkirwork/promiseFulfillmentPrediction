import pandas as pd
import psycopg2
from src.config import DB_CONFIG

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def load_promise_data():
    query = """
    SELECT 
        promise_id,
        ticket_id,
        client_id,
        promised_amount,
        promise_days,
        late_days,
        remaining_principal,
        interest_rate,
        credit_product_type,
        client_age,
        agent_experience_days,
        kept_label,
        paid_in_4d,
        promise_date
    FROM ml_promise_features_v1
    WHERE 
        promised_amount > 0
        AND interest_rate >= 0
        AND client_age >= 18 AND client_age <= 90
        AND kept_label IS NOT NULL
    ORDER BY promise_date
    """
    
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df

def load_unscored_promises():
    query = """
    SELECT 
        promise_id,
        ticket_id,
        client_id,
        promised_amount,
        promise_days,
        late_days,
        remaining_principal,
        interest_rate,
        credit_product_type,
        client_age,
        agent_experience_days
    FROM ml_promise_features_v1
    WHERE 
        promised_amount > 0
        AND interest_rate >= 0
        AND client_age >= 18 AND client_age <= 90
        AND promise_id NOT IN (SELECT promise_id FROM promise_scores)
    ORDER BY promise_date
    """
    
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df

