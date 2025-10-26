CREATE DATABASE promise_fulfillment;


CREATE TABLE ml_promise_features_v1 (
    promise_id SERIAL PRIMARY KEY,
    ticket_id INTEGER NOT NULL,
    client_id INTEGER NOT NULL,
    promised_amount NUMERIC(15, 2) NOT NULL,
    promise_days INTEGER NOT NULL,
    late_days INTEGER NOT NULL,
    remaining_principal NUMERIC(15, 2) NOT NULL,
    interest_rate NUMERIC(5, 2) NOT NULL,
    credit_product_type VARCHAR(50) NOT NULL,
    client_age INTEGER NOT NULL,
    agent_experience_days INTEGER NOT NULL,
    kept_label INTEGER DEFAULT 0,
    paid_in_4d NUMERIC(15, 2) DEFAULT 0,
    promise_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_promise_client ON ml_promise_features_v1(client_id);
CREATE INDEX idx_promise_date ON ml_promise_features_v1(promise_date);
CREATE INDEX idx_kept_label ON ml_promise_features_v1(kept_label);

CREATE TABLE promise_scores (
    score_id SERIAL PRIMARY KEY,
    promise_id INTEGER UNIQUE NOT NULL,
    p_kept NUMERIC(5, 4) NOT NULL,
    class_label INTEGER NOT NULL,
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(20) NOT NULL
);

CREATE INDEX idx_scores_promise ON promise_scores(promise_id);
CREATE INDEX idx_scores_date ON promise_scores(scored_at);

