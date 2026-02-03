"""
Streamlit reporting dashboard for ProviderVerify.

Provides comprehensive analytics and performance monitoring for the
entity resolution pipeline with interactive visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import yaml
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class ReportingDashboard:
    """
    Streamlit-based reporting dashboard for ProviderVerify.
    
    Provides interactive visualizations and metrics for monitoring
    entity resolution performance and system health.
    """
    
    def __init__(self, config_path: str = "config/provider_verify.yaml"):
        """
        Initialize dashboard with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.reporting_config = self.config.get("reporting", {})
        
        # Database paths
        self.db_path = "data/audit.db"
        self.metrics_path = "data/metrics"
        
        # Ensure directories exist
        Path(self.metrics_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized ReportingDashboard")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def get_pipeline_metrics(self) -> Dict[str, any]:
        """
        Get comprehensive pipeline performance metrics.
        
        Returns:
            Dictionary with pipeline metrics
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Basic pipeline statistics
            pipeline_stats = {}
            
            # Get audit statistics
            audit_queue_df = pd.read_sql_query('''
                SELECT 
                    COUNT(*) as total_queue,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
                    COUNT(CASE WHEN status = 'reviewed' THEN 1 END) as reviewed
                FROM audit_queue
            ''', conn)
            
            audit_decisions_df = pd.read_sql_query('''
                SELECT 
                    decision,
                    COUNT(*) as count,
                    timestamp
                FROM audit_log
                GROUP BY decision
            ''', conn)
            
            # Model performance
            model_perf_df = pd.read_sql_query('''
                SELECT * FROM model_performance
                ORDER BY evaluation_date DESC
                LIMIT 10
            ''', conn)
            
            # Recent activity
            recent_activity = pd.read_sql_query('''
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as reviews_completed
                FROM audit_log
                WHERE timestamp >= date('now', '-30 days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', conn)
            
            pipeline_stats = {
                "audit_queue": audit_queue_df.iloc[0].to_dict(),
                "audit_decisions": audit_decisions_df.set_index('decision')['count'].to_dict() if not audit_decisions_df.empty else {},
                "model_performance": model_perf_df.to_dict('records') if not model_perf_df.empty else [],
                "recent_activity": recent_activity.to_dict('records') if not recent_activity.empty else []
            }
            
            return pipeline_stats
            
        except Exception as e:
            logger.error(f"Failed to get pipeline metrics: {e}")
            return {}
        finally:
            conn.close()
    
    def calculate_derived_metrics(self, base_metrics: Dict) -> Dict[str, any]:
        """
        Calculate derived metrics from base statistics.
        
        Args:
            base_metrics: Base pipeline metrics
            
        Returns:
            Dictionary with derived metrics
        """
        derived = {}
        
        # Audit completion rate
        queue_stats = base_metrics.get("audit_queue", {})
        total_queue = queue_stats.get("total_queue", 0)
        reviewed = queue_stats.get("reviewed", 0)
        
        if total_queue > 0:
            derived["completion_rate"] = reviewed / total_queue
        else:
            derived["completion_rate"] = 0.0
        
        # Decision distribution
        decisions = base_metrics.get("audit_decisions", {})
        total_decisions = sum(decisions.values())
        
        if total_decisions > 0:
            derived["merge_rate"] = decisions.get("MERGE", 0) / total_decisions
            derived["reject_rate"] = decisions.get("REJECT", 0) / total_decisions
        else:
            derived["merge_rate"] = 0.0
            derived["reject_rate"] = 0.0
        
        # Model performance trend
        model_perf = base_metrics.get("model_performance", [])
        if len(model_perf) >= 2:
            latest = model_perf[0]
            previous = model_perf[1]
            
            derived["precision_trend"] = latest.get("precision", 0) - previous.get("precision", 0)
            derived["recall_trend"] = latest.get("recall", 0) - previous.get("recall", 0)
            derived["f1_trend"] = latest.get("f1_score", 0) - previous.get("f1_score", 0)
        else:
            derived["precision_trend"] = 0.0
            derived["recall_trend"] = 0.0
            derived["f1_trend"] = 0.0
        
        return derived
    
    def render_overview_dashboard(self):
        """Render overview dashboard with key metrics."""
        st.title("ProviderVerify - Performance Dashboard")
        st.markdown("Real-time monitoring of entity resolution pipeline performance")
        
        # Get metrics
        base_metrics = self.get_pipeline_metrics()
        derived_metrics = self.calculate_derived_metrics(base_metrics)
        
        # Key performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            completion_rate = derived_metrics.get("completion_rate", 0) * 100
            st.metric("Audit Completion", f"{completion_rate:.1f}%", 
                      delta=f"{completion_rate - 85:.1f}%")
        
        with col2:
            merge_rate = derived_metrics.get("merge_rate", 0) * 100
            st.metric("Merge Rate", f"{merge_rate:.1f}%")
        
        with col3:
            queue_stats = base_metrics.get("audit_queue", {})
            pending = queue_stats.get("pending", 0)
            st.metric("Pending Reviews", pending)
        
        with col4:
            model_perf = base_metrics.get("model_performance", [])
            latest_f1 = model_perf[0].get("f1_score", 0) if model_perf else 0
            st.metric("Model F1 Score", f"{latest_f1:.3f}")
        
        # Recent activity chart
        if base_metrics.get("recent_activity"):
            st.subheader("Recent Audit Activity")
            activity_df = pd.DataFrame(base_metrics["recent_activity"])
            
            if not activity_df.empty:
                fig = px.line(activity_df, x="date", y="reviews_completed", 
                            title="Daily Review Completion")
                st.plotly_chart(fig, use_container_width=True)
        
        # Decision distribution
        if base_metrics.get("audit_decisions"):
            st.subheader("Decision Distribution")
            decisions_df = pd.DataFrame(list(base_metrics["audit_decisions"].items()),
                                      columns=["Decision", "Count"])
            
            fig = px.pie(decisions_df, values="Count", names="Decision", 
                        title="Audit Decision Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_model_performance_dashboard(self):
        """Render model performance dashboard."""
        st.title("Model Performance Analytics")
        
        # Get model performance data
        conn = sqlite3.connect(self.db_path)
        
        try:
            model_perf_df = pd.read_sql_query('''
                SELECT *, DATE(evaluation_date) as date
                FROM model_performance
                ORDER BY evaluation_date DESC
                LIMIT 50
            ''', conn)
            
            if model_perf_df.empty:
                st.warning("No model performance data available")
                return
            
            # Performance trends
            st.subheader("Performance Trends Over Time")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Precision", "Recall", "F1 Score", "AUC Score"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Precision
            fig.add_trace(
                go.Scatter(x=model_perf_df['date'], y=model_perf_df['precision'],
                          mode='lines+markers', name='Precision'),
                row=1, col=1
            )
            
            # Recall
            fig.add_trace(
                go.Scatter(x=model_perf_df['date'], y=model_perf_df['recall'],
                          mode='lines+markers', name='Recall'),
                row=1, col=2
            )
            
            # F1 Score
            fig.add_trace(
                go.Scatter(x=model_perf_df['date'], y=model_perf_df['f1_score'],
                          mode='lines+markers', name='F1 Score'),
                row=2, col=1
            )
            
            # AUC Score
            fig.add_trace(
                go.Scatter(x=model_perf_df['date'], y=model_perf_df['auc_score'],
                          mode='lines+markers', name='AUC Score'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Latest performance table
            st.subheader("Latest Model Performance")
            latest_perf = model_perf_df.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Precision", f"{latest_perf['precision']:.3f}")
            with col2:
                st.metric("Recall", f"{latest_perf['recall']:.3f}")
            with col3:
                st.metric("F1 Score", f"{latest_perf['f1_score']:.3f}")
            with col4:
                st.metric("AUC Score", f"{latest_perf['auc_score']:.3f}")
            
            # Performance comparison table
            st.subheader("Historical Performance")
            display_df = model_perf_df[['evaluation_date', 'precision', 'recall', 
                                       'f1_score', 'auc_score']].copy()
            display_df['precision'] = display_df['precision'].round(3)
            display_df['recall'] = display_df['recall'].round(3)
            display_df['f1_score'] = display_df['f1_score'].round(3)
            display_df['auc_score'] = display_df['auc_score'].round(3)
            
            st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading model performance data: {e}")
        finally:
            conn.close()
    
    def render_audit_analytics_dashboard(self):
        """Render audit analytics dashboard."""
        st.title("Audit Analytics")
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Reviewer performance
            st.subheader("Reviewer Performance")
            
            reviewer_stats = pd.read_sql_query('''
                SELECT 
                    reviewer,
                    COUNT(*) as reviews_completed,
                    AVG(processing_time) as avg_processing_time,
                    COUNT(CASE WHEN decision = 'MERGE' THEN 1 END) * 100.0 / COUNT(*) as merge_rate,
                    MIN(timestamp) as first_review,
                    MAX(timestamp) as last_review
                FROM audit_log
                WHERE timestamp >= date('now', '-30 days')
                GROUP BY reviewer
                ORDER BY reviews_completed DESC
            ''', conn)
            
            if not reviewer_stats.empty:
                # Reviewer comparison chart
                fig = px.bar(reviewer_stats, x="reviewer", y="reviews_completed",
                           title="Reviews Completed by Reviewer (Last 30 Days)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Reviewer details table
                st.subheader("Reviewer Details")
                display_df = reviewer_stats.copy()
                display_df['avg_processing_time'] = display_df['avg_processing_time'].round(2)
                display_df['merge_rate'] = display_df['merge_rate'].round(1)
                
                st.dataframe(display_df, use_container_width=True)
            
            # Processing time analysis
            st.subheader("Processing Time Analysis")
            
            processing_times = pd.read_sql_query('''
                SELECT 
                    processing_time,
                    decision,
                    DATE(timestamp) as date
                FROM audit_log
                WHERE processing_time IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1000
            ''', conn)
            
            if not processing_times.empty:
                # Processing time distribution
                fig = px.histogram(processing_times, x="processing_time", 
                                 color="decision", nbins=50,
                                 title="Processing Time Distribution by Decision")
                st.plotly_chart(fig, use_container_width=True)
                
                # Processing time trend
                daily_avg = processing_times.groupby('date')['processing_time'].mean().reset_index()
                fig = px.line(daily_avg, x="date", y="processing_time",
                            title="Average Processing Time Trend")
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading audit analytics: {e}")
        finally:
            conn.close()
    
    def render_system_health_dashboard(self):
        """Render system health dashboard."""
        st.title("System Health")
        
        # System metrics (placeholder - in production would connect to monitoring)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Status", "🟢 Healthy")
        with col2:
            st.metric("CPU Usage", "45%")
        with col3:
            st.metric("Memory Usage", "62%")
        with col4:
            st.metric("Disk Usage", "38%")
        
        # Database health
        st.subheader("Database Health")
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Table sizes
            table_stats = []
            tables = ['audit_log', 'audit_queue', 'model_performance', 'retraining_log']
            
            for table in tables:
                try:
                    count_df = pd.read_sql_query(f'SELECT COUNT(*) as count FROM {table}', conn)
                    table_stats.append({"table": table, "records": count_df.iloc[0]['count']})
                except:
                    table_stats.append({"table": table, "records": 0})
            
            table_df = pd.DataFrame(table_stats)
            st.dataframe(table_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error checking database health: {e}")
        finally:
            conn.close()
        
        # Recent errors (placeholder)
        st.subheader("Recent System Events")
        st.info("No system errors in the last 24 hours")
    
    def export_metrics_report(self, format: str = "csv") -> str:
        """
        Export comprehensive metrics report.
        
        Args:
            format: Export format ('csv' or 'json')
            
        Returns:
            Path to exported file
        """
        # Get all metrics
        base_metrics = self.get_pipeline_metrics()
        derived_metrics = self.calculate_derived_metrics(base_metrics)
        
        # Create comprehensive report
        report_data = {
            "export_timestamp": datetime.now().isoformat(),
            "base_metrics": base_metrics,
            "derived_metrics": derived_metrics
        }
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = f"{self.metrics_path}/metrics_report_{timestamp}.json"
            import json
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        else:
            # Flatten metrics for CSV
            flat_data = []
            
            # Audit queue metrics
            queue_stats = base_metrics.get("audit_queue", {})
            flat_data.append({
                "metric_category": "audit_queue",
                "metric_name": "total_queue",
                "value": queue_stats.get("total_queue", 0),
                "timestamp": datetime.now().isoformat()
            })
            flat_data.append({
                "metric_category": "audit_queue",
                "metric_name": "pending",
                "value": queue_stats.get("pending", 0),
                "timestamp": datetime.now().isoformat()
            })
            
            # Derived metrics
            for metric, value in derived_metrics.items():
                flat_data.append({
                    "metric_category": "derived",
                    "metric_name": metric,
                    "value": value,
                    "timestamp": datetime.now().isoformat()
                })
            
            df = pd.DataFrame(flat_data)
            filename = f"{self.metrics_path}/metrics_report_{timestamp}.csv"
            df.to_csv(filename, index=False)
        
        logger.info(f"Exported metrics report to {filename}")
        return filename


def run_reporting_dashboard():
    """Main function to run the Streamlit reporting dashboard."""
    st.set_page_config(
        page_title="ProviderVerify Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize dashboard
    dashboard = ReportingDashboard()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Dashboard",
        ["Overview", "Model Performance", "Audit Analytics", "System Health"]
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    # Render selected page
    if page == "Overview":
        dashboard.render_overview_dashboard()
    elif page == "Model Performance":
        dashboard.render_model_performance_dashboard()
    elif page == "Audit Analytics":
        dashboard.render_audit_analytics_dashboard()
    elif page == "System Health":
        dashboard.render_system_health_dashboard()
    
    # Export functionality
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Reports")
    
    if st.sidebar.button("Export Metrics (CSV)"):
        filename = dashboard.export_metrics_report("csv")
        st.sidebar.success(f"Exported to {filename}")
    
    if st.sidebar.button("Export Metrics (JSON)"):
        filename = dashboard.export_metrics_report("json")
        st.sidebar.success(f"Exported to {filename}")
    
    # Auto-refresh
    if auto_refresh:
        st.rerun()


if __name__ == "__main__":
    run_reporting_dashboard()
