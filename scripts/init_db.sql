-- PostgreSQL initialization script for ProviderVerify

-- Create database and user
CREATE DATABASE provider_verify;
CREATE USER provider_user WITH ENCRYPTED PASSWORD 'provider_pass';
GRANT ALL PRIVILEGES ON DATABASE provider_verify TO provider_user;

-- Connect to the provider_verify database
\c provider_verify;

-- Create audit schema
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS reporting;
CREATE SCHEMA IF NOT EXISTS config;

-- Set default permissions
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT ALL ON TABLES TO provider_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA reporting GRANT ALL ON TABLES TO provider_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA config GRANT ALL ON TABLES TO provider_user;

-- Audit tables
CREATE TABLE IF NOT EXISTS audit.audit_log (
    audit_id SERIAL PRIMARY KEY,
    pair_id VARCHAR(255) NOT NULL UNIQUE,
    record_id_1 INTEGER NOT NULL,
    record_id_2 INTEGER NOT NULL,
    decision VARCHAR(50) NOT NULL CHECK (decision IN ('MERGE', 'REJECT')),
    comment TEXT,
    reviewer VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hybrid_score DECIMAL(5,4),
    deterministic_score DECIMAL(5,4),
    ml_probability DECIMAL(5,4),
    processing_time DECIMAL(8,3)
);

CREATE TABLE IF NOT EXISTS audit.audit_queue (
    queue_id SERIAL PRIMARY KEY,
    pair_id VARCHAR(255) NOT NULL UNIQUE,
    record_id_1 INTEGER NOT NULL,
    record_id_2 INTEGER NOT NULL,
    hybrid_score DECIMAL(5,4),
    deterministic_score DECIMAL(5,4),
    ml_probability DECIMAL(5,4),
    block_key VARCHAR(255),
    strategy VARCHAR(100),
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'reviewed', 'skipped')),
    priority INTEGER DEFAULT 0,
    processing_time DECIMAL(8,3)
);

-- Model performance tables
CREATE TABLE IF NOT EXISTS reporting.model_performance (
    performance_id SERIAL PRIMARY KEY,
    model_version VARCHAR(100) NOT NULL,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_score DECIMAL(5,4),
    total_pairs INTEGER,
    true_positives INTEGER,
    false_positives INTEGER,
    false_negatives INTEGER,
    true_negatives INTEGER
);

CREATE TABLE IF NOT EXISTS reporting.pipeline_metrics (
    metric_id SERIAL PRIMARY KEY,
    pipeline_run_id VARCHAR(100),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    metric_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Retraining log
CREATE TABLE IF NOT EXISTS audit.retraining_log (
    retraining_id SERIAL PRIMARY KEY,
    retraining_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    old_model_version VARCHAR(100),
    new_model_version VARCHAR(100),
    training_samples INTEGER,
    validation_samples INTEGER,
    training_time DECIMAL(8,3),
    old_performance JSONB,
    new_performance JSONB,
    improvement_metrics JSONB
);

-- Configuration tables
CREATE TABLE IF NOT EXISTS config.pipeline_config (
    config_id SERIAL PRIMARY KEY,
    config_name VARCHAR(100) NOT NULL UNIQUE,
    config_version VARCHAR(50) DEFAULT '1.0',
    config_data JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit.audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_decision ON audit.audit_log(decision);
CREATE INDEX IF NOT EXISTS idx_audit_queue_status ON audit.audit_queue(status);
CREATE INDEX IF NOT EXISTS idx_audit_queue_priority ON audit.audit_queue(priority DESC);
CREATE INDEX IF NOT EXISTS idx_model_performance_date ON reporting.model_performance(evaluation_date DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_timestamp ON reporting.pipeline_metrics(metric_timestamp DESC);

-- Create views for common queries
CREATE OR REPLACE VIEW audit.pending_reviews AS
SELECT 
    queue_id,
    pair_id,
    record_id_1,
    record_id_2,
    hybrid_score,
    deterministic_score,
    ml_probability,
    added_date
FROM audit.audit_queue 
WHERE status = 'pending'
ORDER BY priority DESC, added_date;

CREATE OR REPLACE VIEW audit.review_statistics AS
SELECT 
    DATE(timestamp) as review_date,
    decision,
    COUNT(*) as review_count,
    AVG(processing_time) as avg_processing_time,
    COUNT(CASE WHEN decision = 'MERGE' THEN 1 END) * 100.0 / COUNT(*) as merge_rate
FROM audit.audit_log
WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(timestamp), decision
ORDER BY review_date DESC;

CREATE OR REPLACE VIEW reporting.model_performance_summary AS
SELECT 
    model_version,
    evaluation_date,
    precision,
    recall,
    f1_score,
    auc_score,
    LAG(precision) OVER (ORDER BY evaluation_date) as prev_precision,
    LAG(recall) OVER (ORDER BY evaluation_date) as prev_recall,
    LAG(f1_score) OVER (ORDER BY evaluation_date) as prev_f1
FROM reporting.model_performance
ORDER BY evaluation_date DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO provider_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA reporting TO provider_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA config TO provider_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO provider_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA reporting TO provider_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA config TO provider_user;

-- Insert default configuration
INSERT INTO config.pipeline_config (config_name, config_data, is_active) VALUES
('default', '{
    "scoring": {
        "weights": {
            "name_exact": 0.30,
            "name_fuzzy": 0.20,
            "affiliation": 0.15,
            "location": 0.25,
            "contact": 0.10
        },
        "thresholds": {
            "auto_merge": 0.85,
            "audit_low": 0.65,
            "audit_high": 0.85
        }
    },
    "blocking": {
        "max_candidates_per_block": 1000
    },
    "audit": {
        "sample_rate": 0.07
    }
}', true)
ON CONFLICT (config_name) DO NOTHING;

COMMIT;
