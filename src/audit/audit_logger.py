"""
Audit logging and feedback management for ProviderVerify.

Handles logging of audit decisions, feedback collection, and model
retraining data preparation.
"""

import sqlite3
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import json

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Manages audit logging and feedback collection for ProviderVerify.
    
    Tracks human review decisions, maintains audit trails, and prepares
    data for model retraining and performance monitoring.
    """
    
    def __init__(self, config_path: str = "config/provider_verify.yaml"):
        """
        Initialize audit logger with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.audit_config = self.config.get("audit", {})
        
        # Database paths
        self.db_path = "data/audit.db"
        self.feedback_path = "data/feedback"
        
        # Ensure directories exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.feedback_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info("Initialized AuditLogger")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _init_database(self):
        """Initialize audit database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_id TEXT NOT NULL,
                record_id_1 INTEGER NOT NULL,
                record_id_2 INTEGER NOT NULL,
                decision TEXT NOT NULL,
                comment TEXT,
                reviewer TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                hybrid_score REAL,
                deterministic_score REAL,
                ml_probability REAL,
                processing_time REAL,
                UNIQUE(pair_id)
            )
        ''')
        
        # Audit queue table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_queue (
                queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_id TEXT NOT NULL UNIQUE,
                record_id_1 INTEGER NOT NULL,
                record_id_2 INTEGER NOT NULL,
                hybrid_score REAL,
                deterministic_score REAL,
                ml_probability REAL,
                block_key TEXT,
                strategy TEXT,
                added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                processing_time REAL
            )
        ''')
        
        # Model performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT,
                evaluation_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                precision REAL,
                recall REAL,
                f1_score REAL,
                auc_score REAL,
                total_pairs INTEGER,
                true_positives INTEGER,
                false_positives INTEGER,
                false_negatives INTEGER,
                true_negatives INTEGER
            )
        ''')
        
        # Retraining log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retraining_log (
                retraining_id INTEGER PRIMARY KEY AUTOINCREMENT,
                retraining_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                old_model_version TEXT,
                new_model_version TEXT,
                training_samples INTEGER,
                validation_samples INTEGER,
                training_time REAL,
                old_performance TEXT,
                new_performance TEXT,
                improvement_metrics TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Initialized audit database")
    
    def add_to_audit_queue(self, pairs_df: pd.DataFrame, priority: int = 0):
        """
        Add candidate pairs to audit queue.
        
        Args:
            pairs_df: DataFrame with candidate pairs
            priority: Priority level (higher = more urgent)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        added_count = 0
        for _, pair in pairs_df.iterrows():
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO audit_queue 
                    (pair_id, record_id_1, record_id_2, hybrid_score, deterministic_score, 
                     ml_probability, block_key, strategy, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [
                    f"{pair['record_id_1']}_{pair['record_id_2']}",
                    pair['record_id_1'],
                    pair['record_id_2'],
                    pair.get('hybrid_score', 0.0),
                    pair.get('deterministic_score', 0.0),
                    pair.get('ml_probability', 0.0),
                    pair.get('block_key', ''),
                    pair.get('strategy', ''),
                    priority
                ])
                added_count += 1
            except Exception as e:
                logger.warning(f"Failed to add pair to queue: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added {added_count} pairs to audit queue")
    
    def record_audit_decision(self, pair_id: str, decision: str, 
                           comment: str = "", reviewer: str = "anonymous",
                           processing_time: float = 0.0):
        """
        Record audit decision and update queue status.
        
        Args:
            pair_id: Unique pair identifier
            decision: Audit decision (MERGE/REJECT)
            comment: Optional comment
            reviewer: Reviewer name
            processing_time: Time taken for decision (seconds)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get pair information from queue
            cursor.execute('''
                SELECT record_id_1, record_id_2, hybrid_score, deterministic_score, ml_probability
                FROM audit_queue WHERE pair_id = ?
            ''', [pair_id])
            
            pair_info = cursor.fetchone()
            if not pair_info:
                logger.warning(f"Pair {pair_id} not found in audit queue")
                conn.close()
                return
            
            # Record audit decision
            cursor.execute('''
                INSERT OR REPLACE INTO audit_log 
                (pair_id, record_id_1, record_id_2, decision, comment, reviewer, 
                 hybrid_score, deterministic_score, ml_probability, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                pair_id, pair_info[0], pair_info[1], decision, comment, reviewer,
                pair_info[2], pair_info[3], pair_info[4], processing_time
            ])
            
            # Update queue status
            cursor.execute('''
                UPDATE audit_queue SET status = 'reviewed' WHERE pair_id = ?
            ''', [pair_id])
            
            conn.commit()
            logger.info(f"Recorded audit decision for pair {pair_id}: {decision}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record audit decision: {e}")
        finally:
            conn.close()
    
    def get_audit_queue(self, status: str = "pending", limit: int = 100) -> pd.DataFrame:
        """
        Get audit queue items.
        
        Args:
            status: Queue status filter
            limit: Maximum number of records
            
        Returns:
            DataFrame with audit queue items
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM audit_queue 
            WHERE status = ?
            ORDER BY priority DESC, added_date
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=[status, limit])
        conn.close()
        
        return df
    
    def get_audit_decisions(self, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get audit decisions within date range.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with audit decisions
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def prepare_training_data(self, min_samples: int = 100) -> pd.DataFrame:
        """
        Prepare training data from audit decisions.
        
        Args:
            min_samples: Minimum number of samples required
            
        Returns:
            DataFrame with training data (features + labels)
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get all audit decisions with features
        query = '''
            SELECT 
                pair_id,
                record_id_1,
                record_id_2,
                CASE WHEN decision = 'MERGE' THEN 1 ELSE 0 END as label,
                hybrid_score,
                deterministic_score,
                ml_probability,
                reviewer,
                timestamp
            FROM audit_log
            ORDER BY timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < min_samples:
            logger.warning(f"Insufficient training data: {len(df)} < {min_samples}")
            return pd.DataFrame()
        
        # Add engineered features (simplified for this example)
        df['score_diff'] = df['hybrid_score'] - df['deterministic_score']
        df['high_confidence'] = (df['hybrid_score'] >= 0.9).astype(int)
        df['low_confidence'] = (df['hybrid_score'] <= 0.1).astype(int)
        
        logger.info(f"Prepared {len(df)} training samples from audit decisions")
        return df
    
    def calculate_audit_metrics(self, days: int = 30) -> Dict[str, any]:
        """
        Calculate audit performance metrics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with audit metrics
        """
        conn = sqlite3.connect(self.db_path)
        
        # Recent decisions
        recent_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        decisions_df = pd.read_sql_query('''
            SELECT decision, COUNT(*) as count
            FROM audit_log
            WHERE timestamp >= ?
            GROUP BY decision
        ''', conn, params=[recent_date])
        
        # Queue statistics
        queue_stats = pd.read_sql_query('''
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
                COUNT(CASE WHEN status = 'reviewed' THEN 1 END) as reviewed
            FROM audit_queue
        ''', conn).iloc[0]
        
        # Reviewer performance
        reviewer_stats = pd.read_sql_query('''
            SELECT 
                reviewer,
                COUNT(*) as reviews_completed,
                AVG(processing_time) as avg_processing_time,
                COUNT(CASE WHEN decision = 'MERGE' THEN 1 END) * 100.0 / COUNT(*) as merge_rate
            FROM audit_log
            WHERE timestamp >= ?
            GROUP BY reviewer
            ORDER BY reviews_completed DESC
        ''', conn, params=[recent_date])
        
        conn.close()
        
        # Calculate metrics
        total_reviews = decisions_df['count'].sum()
        merge_count = decisions_df[decisions_df['decision'] == 'MERGE']['count'].sum() if not decisions_df.empty else 0
        reject_count = decisions_df[decisions_df['decision'] == 'REJECT']['count'].sum() if not decisions_df.empty else 0
        
        metrics = {
            "period_days": days,
            "total_reviews": total_reviews,
            "merge_count": merge_count,
            "reject_count": reject_count,
            "merge_rate": merge_count / total_reviews if total_reviews > 0 else 0,
            "queue_statistics": queue_stats.to_dict(),
            "reviewer_performance": reviewer_stats.to_dict('records'),
            "decision_distribution": decisions_df.set_index('decision')['count'].to_dict() if not decisions_df.empty else {}
        }
        
        return metrics
    
    def log_model_performance(self, model_version: str, metrics: Dict[str, float]):
        """
        Log model performance metrics.
        
        Args:
            model_version: Model version identifier
            metrics: Performance metrics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance 
            (model_version, precision, recall, f1_score, auc_score, 
             total_pairs, true_positives, false_positives, false_negatives, true_negatives)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [
            model_version,
            metrics.get('precision', 0.0),
            metrics.get('recall', 0.0),
            metrics.get('f1_score', 0.0),
            metrics.get('auc_score', 0.0),
            metrics.get('total_pairs', 0),
            metrics.get('true_positives', 0),
            metrics.get('false_positives', 0),
            metrics.get('false_negatives', 0),
            metrics.get('true_negatives', 0)
        ])
        
        conn.commit()
        conn.close()
        
        logger.info(f"Logged performance metrics for model {model_version}")
    
    def log_retraining_event(self, old_model: str, new_model: str, 
                           training_info: Dict[str, any]):
        """
        Log model retraining event.
        
        Args:
            old_model: Old model version
            new_model: New model version
            training_info: Training information dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO retraining_log 
            (old_model_version, new_model_version, training_samples, validation_samples,
             training_time, old_performance, new_performance, improvement_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', [
            old_model,
            new_model,
            training_info.get('training_samples', 0),
            training_info.get('validation_samples', 0),
            training_info.get('training_time', 0.0),
            json.dumps(training_info.get('old_performance', {})),
            json.dumps(training_info.get('new_performance', {})),
            json.dumps(training_info.get('improvement_metrics', {}))
        ])
        
        conn.commit()
        conn.close()
        
        logger.info(f"Logged retraining event: {old_model} -> {new_model}")
    
    def export_feedback_data(self, output_path: Optional[str] = None) -> str:
        """
        Export all feedback data for external analysis.
        
        Args:
            output_path: Output file path (optional)
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.feedback_path}/audit_export_{timestamp}.csv"
        
        # Get all audit decisions
        decisions_df = self.get_audit_decisions()
        
        # Save to CSV
        decisions_df.to_csv(output_path, index=False)
        
        logger.info(f"Exported audit feedback to {output_path}")
        return output_path
    
    def should_retrain_model(self, threshold: int = 100) -> bool:
        """
        Check if model should be retrained based on new feedback.
        
        Args:
            threshold: Minimum number of new decisions required
            
        Returns:
            True if retraining is recommended
        """
        # Get count of recent decisions (since last retraining)
        conn = sqlite3.connect(self.db_path)
        
        cursor.execute('''
            SELECT MAX(retraining_date) FROM retraining_log
        ''')
        last_retraining = cursor.fetchone()[0]
        
        if last_retraining:
            cursor.execute('''
                SELECT COUNT(*) FROM audit_log WHERE timestamp > ?
            ''', [last_retraining])
        else:
            cursor.execute('SELECT COUNT(*) FROM audit_log')
        
        new_decisions = cursor.fetchone()[0]
        conn.close()
        
        should_retrain = new_decisions >= threshold
        
        if should_retrain:
            logger.info(f"Model retraining recommended: {new_decisions} new decisions")
        else:
            logger.info(f"Model retraining not needed: {new_decisions}/{threshold} new decisions")
        
        return should_retrain


def create_audit_logger(config_path: str = "config/provider_verify.yaml") -> AuditLogger:
    """
    Convenience function to create audit logger.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized audit logger
    """
    return AuditLogger(config_path)
