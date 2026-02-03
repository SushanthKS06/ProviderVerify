"""
Streamlit audit UI for ProviderVerify.

Provides human review interface for borderline provider matches with
side-by-side comparison, decision capture, and feedback logging.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import yaml
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditUI:
    """
    Streamlit-based user interface for provider match auditing.
    
    Enables human reviewers to evaluate borderline matches and provide
    feedback to improve model performance.
    """
    
    def __init__(self, config_path: str = "config/provider_verify.yaml"):
        """
        Initialize audit UI with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.audit_config = self.config.get("audit", {})
        
        # Initialize audit database
        self.db_path = "data/audit.db"
        self._init_audit_db()
        
        logger.info("Initialized AuditUI")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _init_audit_db(self):
        """Initialize audit database for storing decisions."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create audit_log table
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
                ml_probability REAL
            )
        ''')
        
        # Create audit_queue table for pending reviews
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_queue (
                queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_id TEXT NOT NULL,
                record_id_1 INTEGER NOT NULL,
                record_id_2 INTEGER NOT NULL,
                hybrid_score REAL,
                deterministic_score REAL,
                ml_probability REAL,
                added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Initialized audit database")
    
    def load_audit_queue(self, limit: int = 50) -> pd.DataFrame:
        """
        Load pending audit queue from database.
        
        Args:
            limit: Maximum number of records to load
            
        Returns:
            DataFrame with pending audit records
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM audit_queue 
            WHERE status = 'pending'
            ORDER BY added_date
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=[limit])
        conn.close()
        
        return df
    
    def add_to_audit_queue(self, pairs_df: pd.DataFrame):
        """
        Add candidate pairs to audit queue.
        
        Args:
            pairs_df: DataFrame with candidate pairs to audit
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for _, pair in pairs_df.iterrows():
            cursor.execute('''
                INSERT INTO audit_queue 
                (pair_id, record_id_1, record_id_2, hybrid_score, deterministic_score, ml_probability)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', [
                f"{pair['record_id_1']}_{pair['record_id_2']}",
                pair['record_id_1'],
                pair['record_id_2'],
                pair.get('hybrid_score', 0.0),
                pair.get('deterministic_score', 0.0),
                pair.get('ml_probability', 0.0)
            ])
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added {len(pairs_df)} pairs to audit queue")
    
    def record_audit_decision(self, pair_id: str, decision: str, 
                           comment: str = "", reviewer: str = "anonymous",
                           scores: Dict = None):
        """
        Record audit decision and update queue status.
        
        Args:
            pair_id: Unique pair identifier
            decision: Audit decision (MERGE/REJECT)
            comment: Optional comment
            reviewer: Reviewer name
            scores: Score information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
            INSERT INTO audit_log 
            (pair_id, record_id_1, record_id_2, decision, comment, reviewer, 
             hybrid_score, deterministic_score, ml_probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [
            pair_id, pair_info[0], pair_info[1], decision, comment, reviewer,
            pair_info[2], pair_info[3], pair_info[4]
        ])
        
        # Update queue status
        cursor.execute('''
            UPDATE audit_queue SET status = 'reviewed' WHERE pair_id = ?
        ''', [pair_id])
        
        conn.commit()
        conn.close()
        
        logger.info(f"Recorded audit decision for pair {pair_id}: {decision}")
    
    def get_audit_statistics(self) -> Dict[str, any]:
        """
        Get audit statistics and performance metrics.
        
        Returns:
            Dictionary with audit statistics
        """
        conn = sqlite3.connect(self.db_path)
        
        # Queue statistics
        queue_stats = pd.read_sql_query('''
            SELECT 
                COUNT(*) as total_pending,
                COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
                COUNT(CASE WHEN status = 'reviewed' THEN 1 END) as reviewed
            FROM audit_queue
        ''', conn).iloc[0].to_dict()
        
        # Decision statistics
        decision_stats = pd.read_sql_query('''
            SELECT 
                decision,
                COUNT(*) as count,
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM audit_log) as percentage
            FROM audit_log
            GROUP BY decision
        ''', conn)
        
        # Recent activity
        recent_activity = pd.read_sql_query('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as reviews_completed
            FROM audit_log
            WHERE timestamp >= date('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 7
        ''', conn)
        
        conn.close()
        
        return {
            "queue_statistics": queue_stats,
            "decision_distribution": decision_stats.set_index('decision').to_dict(),
            "recent_activity": recent_activity.to_dict('records')
        }
    
    def render_audit_interface(self, providers_df: pd.DataFrame):
        """
        Render the main audit interface.
        
        Args:
            providers_df: DataFrame with provider records
        """
        st.title("ProviderVerify - Audit Interface")
        st.markdown("Review borderline provider matches and provide feedback")
        
        # Load audit queue
        audit_queue = self.load_audit_queue()
        
        if audit_queue.empty:
            st.warning("No pending items in audit queue")
            return
        
        # Sidebar statistics
        with st.sidebar:
            st.header("Audit Statistics")
            stats = self.get_audit_statistics()
            
            st.metric("Pending Reviews", stats["queue_statistics"]["pending"])
            st.metric("Completed Reviews", stats["queue_statistics"]["reviewed"])
            
            if not stats["decision_distribution"]["count"].empty:
                st.subheader("Decision Distribution")
                for decision, count in stats["decision_distribution"]["count"].items():
                    st.write(f"{decision}: {count}")
        
        # Main audit interface
        st.header("Pending Reviews")
        
        # Select pair to review
        pair_options = [f"Pair {i+1}: {row['record_id_1']} ↔ {row['record_id_2']}" 
                       for i, row in audit_queue.iterrows()]
        
        if pair_options:
            selected_pair_idx = st.selectbox("Select pair to review:", 
                                          range(len(pair_options)), 
                                          format_func=lambda x: pair_options[x])
            
            if selected_pair_idx is not None:
                self._render_pair_review(audit_queue.iloc[selected_pair_idx], providers_df)
    
    def _render_pair_review(self, pair_info: pd.Series, providers_df: pd.DataFrame):
        """Render detailed pair review interface."""
        pair_id = pair_info['pair_id']
        record_id_1 = pair_info['record_id_1']
        record_id_2 = pair_info['record_id_2']
        
        # Get provider records
        rec1 = providers_df.loc[record_id_1].to_dict()
        rec2 = providers_df.loc[record_id_2].to_dict()
        
        # Display scores
        st.subheader("Similarity Scores")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Hybrid Score", f"{pair_info['hybrid_score']:.3f}")
        with col2:
            st.metric("Deterministic", f"{pair_info['deterministic_score']:.3f}")
        with col3:
            st.metric("ML Probability", f"{pair_info['ml_probability']:.3f}")
        
        # Side-by-side comparison
        st.subheader("Provider Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Provider 1**")
            self._render_provider_card(rec1, record_id_1)
        
        with col2:
            st.write("**Provider 2**")
            self._render_provider_card(rec2, record_id_2)
        
        # Decision interface
        st.subheader("Audit Decision")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🤝 MERGE", type="primary", key="merge_btn"):
                self._record_decision(pair_id, "MERGE")
                st.rerun()
        
        with col2:
            if st.button("❌ REJECT", type="secondary", key="reject_btn"):
                self._record_decision(pair_id, "REJECT")
                st.rerun()
        
        with col3:
            comment = st.text_input("Comment (optional):")
            reviewer = st.text_input("Reviewer name:", value="anonymous")
        
        # Store comment and reviewer in session state
        if comment:
            st.session_state[f"comment_{pair_id}"] = comment
        if reviewer:
            st.session_state[f"reviewer_{pair_id}"] = reviewer
    
    def _render_provider_card(self, provider: Dict, record_id: int):
        """Render individual provider information card."""
        # Key fields to display
        fields = [
            ("Name", "norm_name"),
            ("First Name", "first_name_norm"),
            ("Last Name", "last_name_norm"),
            ("Affiliation", "norm_affiliation"),
            ("Address", "address_line_1_norm"),
            ("City", "city_norm"),
            ("State", "state_norm"),
            ("ZIP", "zip_norm"),
            ("Phone", "phone_norm"),
            ("Email", "email_norm"),
            ("Source", "source")
        ]
        
        for label, field in fields:
            value = provider.get(field, "")
            if value:
                st.write(f"**{label}:** {value}")
        
        st.write(f"**Record ID:** {record_id}")
    
    def _record_decision(self, pair_id: str, decision: str):
        """Record audit decision with comment and reviewer."""
        comment = st.session_state.get(f"comment_{pair_id}", "")
        reviewer = st.session_state.get(f"reviewer_{pair_id}", "anonymous")
        
        self.record_audit_decision(pair_id, decision, comment, reviewer)
        
        # Clear session state
        if f"comment_{pair_id}" in st.session_state:
            del st.session_state[f"comment_{pair_id}"]
        if f"reviewer_{pair_id}" in st.session_state:
            del st.session_state[f"reviewer_{pair_id}"]
        
        st.success(f"Recorded decision: {decision}")
    
    def render_feedback_dashboard(self):
        """Render feedback analysis dashboard."""
        st.title("Audit Feedback Dashboard")
        
        stats = self.get_audit_statistics()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", stats["queue_statistics"]["reviewed"])
        with col2:
            st.metric("Pending", stats["queue_statistics"]["pending"])
        with col3:
            merge_rate = 0
            if not stats["decision_distribution"]["count"].empty and "MERGE" in stats["decision_distribution"]["count"]:
                merge_rate = stats["decision_distribution"]["count"]["MERGE"] / stats["queue_statistics"]["reviewed"] * 100
            st.metric("Merge Rate", f"{merge_rate:.1f}%")
        with col4:
            st.metric("Completion Rate", 
                     f"{stats['queue_statistics']['reviewed'] / (stats['queue_statistics']['pending'] + stats['queue_statistics']['reviewed']) * 100:.1f}%")
        
        # Decision distribution chart
        if not stats["decision_distribution"]["count"].empty:
            st.subheader("Decision Distribution")
            decision_df = pd.DataFrame(stats["decision_distribution"])
            st.bar_chart(decision_df.set_index("decision")["count"])
        
        # Recent activity
        if stats["recent_activity"]:
            st.subheader("Recent Activity (Last 7 Days)")
            activity_df = pd.DataFrame(stats["recent_activity"])
            if not activity_df.empty:
                st.line_chart(activity_df.set_index("date")["reviews_completed"])
    
    def export_audit_data(self) -> pd.DataFrame:
        """
        Export all audit decisions for model retraining.
        
        Returns:
            DataFrame with audit decisions for ML training
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                pair_id,
                record_id_1,
                record_id_2,
                CASE WHEN decision = 'MERGE' THEN 1 ELSE 0 END as label,
                hybrid_score,
                deterministic_score,
                ml_probability,
                comment,
                reviewer,
                timestamp
            FROM audit_log
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"Exported {len(df)} audit decisions for training")
        return df


def run_audit_ui():
    """Main function to run the Streamlit audit UI."""
    st.set_page_config(
        page_title="ProviderVerify Audit",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize audit UI
    audit_ui = AuditUI()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", ["Audit Reviews", "Feedback Dashboard"])
    
    # Load provider data (in production, this would come from your data store)
    # For demo purposes, we'll show a message
    if page == "Audit Reviews":
        st.info("Provider data loading... In production, this connects to your provider database.")
        audit_ui.render_audit_interface(pd.DataFrame())  # Empty DataFrame for demo
    elif page == "Feedback Dashboard":
        audit_ui.render_feedback_dashboard()


if __name__ == "__main__":
    run_audit_ui()
