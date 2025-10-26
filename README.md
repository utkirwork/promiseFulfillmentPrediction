# Promise Fulfillment Prediction Model

Bu loyiha **Soft Collection** tizimi uchun ishlab chiqilgan ML modeldir. Model mijozdan to'lov va'dasi (promise) olingandan keyin, keyingi **4 kun** ichida haqiqiy to'lov bo'ladimi-yo'qligini bashorat qiladi.

## Xususiyatlar

- **XGBoost** algoritmi asosida binar klassifikatsiya
- **PR-AUC** metrikasi asosida optimallashtirilgan
- **9 ta feature** ishlatiladi (va'da summasidan agent tajribasigacha)
- Docker Compose orqali oson setup
- PostgreSQL baza bilan integratsiya

## Loyiha Strukturasi

```
.
├── docker-compose.yml          # PostgreSQL + Jupyter
├── database/
│   └── init.sql               # Baza jadvallari
├── src/
│   ├── config.py              # Konfiguratsiya
│   ├── data_loader.py         # Ma'lumot yuklash
│   ├── feature_engineering.py # Feature tayyorlash
│   ├── train.py               # Model tayyorlash
│   ├── predict.py             # Batch scoring
│   └── evaluate.py            # Model baholash
├── generate_sample_data.py     # Sintetik ma'lumot yaratish
└── requirements.txt           # Python kutubxonalari
```

## O'rnatish (Setup)

### 1. Kerakli narsalar

- Docker va Docker Compose
- Python 3.8+

### 2. Loyihani yuklash va o'rnatish

```bash
git clone <repository>
cd PromiseFulfillment

docker-compose up -d
```

Bu quyidagi xizmatlarni ko'taradi:
- **PostgreSQL** (port 5432)
- **Jupyter Notebook** (port 8888)

### 3. Python kutubxonalarini o'rnatish

```bash
pip install -r requirements.txt
```

## Ishlatish

### 1. Sintetik ma'lumot yaratish

Modelni sinash uchun test ma'lumotlari yarating:

```bash
python generate_sample_data.py 3000
```

Bu 3000 ta sintetik yozuv yaratadi va bazaga yozadi.

### 2. Modelni tayyorlash (Training)

```bash
python src/train.py
```

Model tayyorlandikdan keyin quyidagi fayllar yaratiladi:
- `models/PFM_v1.json` - XGBoost model
- `models/PFM_v1_encoders.pkl` - Feature encoders
- `models/PFM_v1_metadata.pkl` - Model metadata

### 3. Prediction (Batch Scoring)

Yangi va'dalarga ehtimol hisoblash:

```bash
python src/predict.py
```

Bu skript:
- Bazadan baholanmay qolgan va'dalarni topadi
- Modelni yuklaydi va baholaydi
- Natijalarni `promise_scores` jadvaliga yozadi

### 4. Modelni baholash

```bash
python src/evaluate.py
```

Bu skript modelning PR-AUC, ROC-AUC va boshqa metrikalarini ko'rsatadi.

## Featurelar (9 ta)

Model quyidagi 8 ta feature'dan foydalanadi:

1. `promised_amount` - Va'da qilingan summa
2. `promise_days` - Va'da muddati (kun)
3. `late_days` - Va'da paytidagi kechikish (DPD)
4. `remaining_principal` - Asosiy qarz qoldig'i
5. `interest_rate` - Kredit foiz stavkasi
6. `credit_product_type` - Kredit turi (ipoteka/iste'mol/avto...)
7. `client_age` - Mijoz yoshi
8. `agent_experience_days` - Agent ish tajribasi (kunlarda)

## Baza Strukturasi

### `ml_promise_features_v1`

Asosiy feature table'ga quyidagi ustunlar kiradi:

- `promise_id` (PK)
- `ticket_id`, `client_id`
- 8 ta feature ustunlari
- `kept_label` (0/1) - target
- `paid_in_4d` - 4 kunda to'langan summa
- `promise_date`
- `created_at`

### `promise_scores`

Model natijalari uchun table:

- `score_id` (PK)
- `promise_id`
- `p_kept` (0-1 orasida ehtimol)
- `class_label` (0/1 prediction)
- `scored_at`
- `model_version`

## Model Konfiguratsiyasi

`src/config.py` faylida quyidagi konfiguratsiyalarni o'zgartirishingiz mumkin:

- **Threshold:** 0.6 (default) - ehtimol chegarasi
- **Temporal split:** 80% train / 20% validation
- **XGBoost params:** max_depth, learning_rate, n_estimators va boshqalar

## Monitoring va Maintenance

### Model performance

Modeli monitoring qilish uchun `src/evaluate.py` ni muntazam ishlating. 

**Retrain trigger:**
- PR-AUC 5% yoki ko'proq pasayganda
- Mavjudlik drift aniqlanganda

### Ma'lumotlar tozalanishi

Model faqat toza ma'lumotlar bilan ishlaydi:

- `promised_amount > 0`
- `interest_rate >= 0`
- `client_age` o'rtasida 18 va 90
- `kept_label IS NOT NULL`

## Natijalarni SQL orqali olish

```sql
SELECT 
    p.promise_id,
    p.promised_amount,
    p.credit_product_type,
    s.p_kept,
    s.class_label,
    s.scored_at
FROM ml_promise_features_v1 p
JOIN promise_scores s ON p.promise_id = s.promise_id
ORDER BY s.scored_at DESC
LIMIT 100;
```

## Yordam

Muammo bo'lsa yoki savol bo'lsa, muallifga murojaat qiling.

