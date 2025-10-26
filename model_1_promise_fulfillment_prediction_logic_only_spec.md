# MODEL 1 — Promise Fulfillment Prediction (Logic-Only, v1 — Only Promise Tickets)

## 1. Maqsad
Bu model **Soft Collection** tizimi uchun ishlab chiqilgan. Operator mijozdan **to‘lov va’dasini** olgan holatlarda (faqat **va’dali ticketlar**) keyingi **4 kun** ichida haqiqiy to‘lov bo‘ladimi-yo‘qligini oldindan baholaydi.

**Natija:** Har bir va’da uchun `p_kept ∈ [0,1]` ehtimol, UI’da **foiz** ko‘rinishida (masalan, **82%**) va rangli band (Green/Yellow/Red).  
**Biznes maqsadlari:** operator ishini ustuvorlashtirish, erta to‘lovni oshirish, kechikish xavfini erta aniqlash.

---

## 2. Qo‘llanish sohasi (Scope)
- **Predmet:** faqat **`ticket_promises`** (yoki `has_promise = true`) yozuvlari.  
- **No-promise ticketlar:** **Model-1 qo‘llanmaydi** (N/A). Ular alohida modellar (Payment Propensity/Reachability) doirasiga kiradi.
- **Vaqt oynasi (label uchun):** va’da sanasidan keyingi **4 kun**.

---

## 3. Ma’lumot manbalari (v1)
- **Promises:** `ticket_promises` — `promise_id`, `ticket_id`, `client_id`, `promised_amount`, `promise_days`, `promise_date`.  
- **Credit snapshot:** `loan_snapshot/credits` — `late_days (DPD)`, `remaining_principal`, `interest_rate`, `credit_product_type`.  
- **Client profile:** `clients` — `client_age`.  
- **Agent profile:** `agents` — `agent_experience_days`.

---

## 4. Target (Label)
- **Asosiy label:** `kept_label ∈ {0,1}` — va’da sanasidan keyingi **4 kun** ichida ≥1 to‘lov bo‘lsa **1**, aks holda **0**.  
- **Yordamchi (train uchun ishlatilmaydi):** `coef_0_1 = min( paid_in_4d / promised_amount , 1 )`.

**Leakage qoidalari (majburiy):**
- Featurelar faqat **va’da olingan paytdagi** ma’lumotlardan quriladi (operator ko‘radigan real kontekst).  
- **Va’dadan keyingi** to‘lovlar, status/condition/event’lar **modelga kiritilmaydi**.  
- **Temporal split:** train → valid → test vaqt bo‘yicha ketma-ketlikda.

---

## 5. Aktiv Featurelar (X) — **v1 minimal (faqat real-time)**
Quyidagi ustunlar **train va production**da aynan bir xil ishlatiladi.

| № | Column | Ta’rif |
|---|--------|--------|
| 1 | `promised_amount` | Va’da qilingan summa |
| 2 | `promise_days` | Va’da muddati (kun) |
| 3 | `late_days` | Va’da paytidagi kechikish (DPD) |
| 4 | `remaining_principal` | Asosiy qarz qoldig‘i |
| 5 | `interest_rate` | Kredit foiz stavkasi |
| 6 | `credit_product_type` | Kredit turi (ipoteka/iste’mol/avto…) |
| 7 | `client_age` | Mijoz yoshi |
| 8 | `agent_experience_days` | Agent ish tajribasi (kunlarda) |

> Eslatma: `client_segment` v1 da mavjud emas (faqat jismoniy mijozlar uchun). Keyingi versiyalarda (v2) segmentatsiya ma’lumotlari paydo bo‘lsa, qo‘shilishi mumkin.

---

## 6. Training Dataset siyosati
- **Kirish:** faqat `has_promise = true` yozuvlar.  
- **Chiqarib tashlanadiganlar:** `promised_amount ≤ 0`, `interest_rate < 0`, mantiqsiz `client_age` ( <18 yoki >90 ).  
- **Categorical kodlash:** `credit_product_type` (one-hot yoki native cat).  
- **Imbalance:** class_weight yoki threshold tuning (PR-AUC optimallashtirish).

---

## 7. Model tanlovi
- **Algoritm:** **XGBoost (binary classifier)** — tez, barqaror, imbalanced setlarda yaxshi.  
- **Asosiy metrika:** **PR-AUC** (asosiy), qo‘shimcha: ROC-AUC, Brier, KS.

---

## 8. Serving (v1)
- **Batch scoring (kunlik):** `ml_promise_features_v1` → `promise_scores` ( `promise_id`, `p_kept`, `class_label`, `scored_at`, `model_version` ).  
- **UI:** `p_kept` **foiz** sifatida (masalan **82%**) va rangli band. Threshold default: `0.6` (konfiguratsion).

---

## 9. Monitoring
- **Performance:** PR-AUC/ROC-AUC oyma-oy; **kalibratsiya** (reliability).  
- **Data drift:** feature distribution (Evidently/BI).  
- **Retrain trigger:** PR-AUC −5% yoki drift > threshold.

---

## 10. Versiyalash
- **Model:** `PFM_v1`  
- **Feature spec:** `FS_PFM_v1` (faqat yuqoridagi 8 ta ustun)  
- **Label spec:** `LBL_PFM_v1` (4 kunlik kept)