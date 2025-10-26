import random
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from src.config import DB_CONFIG

credit_product_types = ['ipoteka', 'iste_mol', 'avto', 'biznes', 'mikro']

def generate_sample_data(n_records=3000):
    print(f"{n_records} ta sintetik ma'lumot yaratilmoqda...")
    
    promises = []
    
    for i in range(n_records):
        promised_amount = random.uniform(50000, 5000000)
        promise_days = random.choice([3, 5, 7, 10, 14, 21, 30])
        late_days = random.randint(0, 180)
        
        remaining_principal = promised_amount * random.uniform(0.5, 3.0)
        interest_rate = random.uniform(12.0, 36.0)
        
        credit_product_type = random.choice(credit_product_types)
        client_age = random.randint(25, 65)
        agent_experience_days = random.randint(30, 1000)
        
        promise_date = datetime.now() - timedelta(days=random.randint(1, 180))
        
        kept = random.random() < 0.4
        
        if kept:
            kept_label = 1
            paid_in_4d = min(promised_amount, promised_amount * random.uniform(0.3, 1.5))
        else:
            kept_label = 0
            paid_in_4d = random.uniform(0, promised_amount * 0.15)
        
        promises.append({
            'ticket_id': 1000 + i,
            'client_id': 500 + (i % 200),
            'promised_amount': round(promised_amount, 2),
            'promise_days': promise_days,
            'late_days': late_days,
            'remaining_principal': round(remaining_principal, 2),
            'interest_rate': round(interest_rate, 2),
            'credit_product_type': credit_product_type,
            'client_age': client_age,
            'agent_experience_days': agent_experience_days,
            'kept_label': kept_label,
            'paid_in_4d': round(paid_in_4d, 2),
            'promise_date': promise_date
        })
    
    df = pd.DataFrame(promises)
    print(f"\nMa'lumotlar tayyorlandi:")
    print(f"  Jami: {len(df)}")
    print(f"  To'lov qilganlar: {df['kept_label'].sum()} ({df['kept_label'].mean()*100:.1f}%)")
    
    return df

def insert_to_database(df):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    print("\nMa'lumotlar bazaga yozilmoqda...")
    
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO ml_promise_features_v1 (
                ticket_id, client_id, promised_amount, promise_days, late_days,
                remaining_principal, interest_rate, credit_product_type,
                client_age, agent_experience_days,
                kept_label, paid_in_4d, promise_date
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """, (
            int(row['ticket_id']), int(row['client_id']),
            float(row['promised_amount']), int(row['promise_days']), int(row['late_days']),
            float(row['remaining_principal']), float(row['interest_rate']), row['credit_product_type'],
            int(row['client_age']), int(row['agent_experience_days']),
            int(row['kept_label']), float(row['paid_in_4d']), row['promise_date']
        ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"{len(df)} ta yozuv bazaga yozildi.")

if __name__ == '__main__':
    import sys
    
    n_records = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    df = generate_sample_data(n_records)
    insert_to_database(df)
    print("\nBajarildi!")

