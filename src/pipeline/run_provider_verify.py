"""
Main pipeline orchestrator for ProviderVerify.

Coordinates the complete entity resolution pipeline from data ingestion
through normalization, blocking, scoring, merging, and reporting.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.s3_loader import S3DataLoader, create_spark_session
from ingestion.schema_validator import validate_provider_data
from normalize.name_normalizer import normalize_provider_names
from normalize.affiliation_normalizer import normalize_provider_affiliations
from normalize.address_normalizer import normalize_provider_addresses
from normalize.config import load_normalization_config
from blocking.block_key_builder import create_blocking_keys
from match.deterministic_rules import apply_deterministic_scoring
from match.scorer import apply_hybrid_scoring
from merge.merger import merge_provider_records
from audit.audit_logger import create_audit_logger
from reporting.dashboard import ReportingDashboard

logger = logging.getLogger(__name__)


class ProviderVerifyPipeline:
    """
    Main pipeline orchestrator for ProviderVerify.
    
    Coordinates all components of the entity resolution pipeline
    with comprehensive logging and error handling.
    """
    
    def __init__(self, config_path: str = "config/provider_verify.yaml"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.spark_session = None
        self.audit_logger = create_audit_logger(config_path)
        self.dashboard = ReportingDashboard(config_path)
        
        # Pipeline state
        self.pipeline_start_time = None
        self.stage_times = {}
        
        logger.info("Initialized ProviderVerify pipeline")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    
    def _start_stage_timer(self, stage_name: str):
        """Start timing for a pipeline stage."""
        self.stage_times[stage_name] = time.time()
        logger.info(f"Starting stage: {stage_name}")
    
    def _end_stage_timer(self, stage_name: str):
        """End timing for a pipeline stage."""
        if stage_name in self.stage_times:
            duration = time.time() - self.stage_times[stage_name]
            logger.info(f"Completed stage: {stage_name} in {duration:.2f} seconds")
    
    def ingest_data(self, input_path: str, source_label: str, 
                   use_spark: bool = False) -> pd.DataFrame:
        """
        Ingest provider data from specified source.
        
        Args:
            input_path: Path to input data
            source_label: Label for data source
            use_spark: Whether to use Spark for processing
            
        Returns:
            DataFrame with ingested data
        """
        self._start_stage_timer("data_ingestion")
        
        try:
            # Initialize Spark if requested
            if use_spark:
                spark_config = self.config.get("processing", {}).get("spark", {})
                self.spark_session = create_spark_session(
                    app_name=spark_config.get("app_name", "ProviderVerify"),
                    master=spark_config.get("master", "local[*]")
                )
            
            # Load data based on input type
            if input_path.startswith("s3://"):
                # Load from S3
                s3_config = self.config.get("data_sources", {})
                loader = S3DataLoader(
                    bucket_name=s3_config.get("bucket_name"),
                    region_name="us-east-1"
                )
                
                # Parse S3 path
                s3_path = input_path.replace("s3://", "")
                bucket, key = s3_path.split("/", 1)
                
                df = loader.load_file(key, spark_session=self.spark_session)
            else:
                # Load from local file
                if input_path.endswith(".csv"):
                    df = pd.read_csv(input_path)
                elif input_path.endswith(".parquet"):
                    df = pd.read_parquet(input_path)
                elif input_path.endswith(".json"):
                    df = pd.read_json(input_path, lines=True)
                else:
                    raise ValueError(f"Unsupported file format: {input_path}")
            
            # Add source label
            df["source"] = source_label
            
            logger.info(f"Ingested {len(df)} records from {input_path}")
            
            self._end_stage_timer("data_ingestion")
            return df
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate provider data against schema.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated and cleaned DataFrame
        """
        self._start_stage_timer("data_validation")
        
        try:
            schema_config = self.config.get("schema", {})
            validated_df, validation_summary = validate_provider_data(df, schema_config)
            
            logger.info(f"Data validation completed: {validation_summary.get('success_rate', 0):.2%} success rate")
            
            self._end_stage_timer("data_validation")
            return validated_df
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize provider data fields.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Normalized DataFrame
        """
        self._start_stage_timer("data_normalization")
        
        try:
            # Load normalization configuration
            norm_config = load_normalization_config(self.config_path)
            
            # Apply normalization
            normalized_df = df.copy()
            
            # Normalize names
            normalized_df = normalize_provider_names(normalized_df, norm_config.get("name", {}))
            
            # Normalize affiliations
            normalized_df = normalize_provider_affiliations(normalized_df, norm_config.get("affiliation", {}))
            
            # Normalize addresses and contact info
            normalized_df = normalize_provider_addresses(normalized_df, norm_config.get("address", {}))
            
            logger.info(f"Data normalization completed for {len(normalized_df)} records")
            
            self._end_stage_timer("data_normalization")
            return normalized_df
            
        except Exception as e:
            logger.error(f"Data normalization failed: {e}")
            raise
    
    def generate_candidates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate candidate pairs using blocking strategies.
        
        Args:
            df: Normalized provider DataFrame
            
        Returns:
            Tuple of (df_with_block_keys, candidate_pairs_df)
        """
        self._start_stage_timer("candidate_generation")
        
        try:
            blocking_config = self.config.get("blocking", {})
            df_with_keys, candidates_df = create_blocking_keys(df, blocking_config)
            
            logger.info(f"Generated {len(candidates_df)} candidate pairs")
            
            self._end_stage_timer("candidate_generation")
            return df_with_keys, candidates_df
            
        except Exception as e:
            logger.error(f"Candidate generation failed: {e}")
            raise
    
    def score_candidates(self, candidates_df: pd.DataFrame, 
                        providers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score candidate pairs using hybrid approach.
        
        Args:
            candidates_df: Candidate pairs DataFrame
            providers_df: Provider records DataFrame
            
        Returns:
            DataFrame with hybrid scores and decisions
        """
        self._start_stage_timer("candidate_scoring")
        
        try:
            # Apply deterministic scoring
            scoring_config = self.config.get("scoring", {})
            scored_pairs_df = apply_deterministic_scoring(
                candidates_df, providers_df, scoring_config
            )
            
            # Apply hybrid scoring
            ml_model_path = "models/provider_match.xgb"  # Default path
            hybrid_df = apply_hybrid_scoring(
                scored_pairs_df, candidates_df, providers_df,
                self.config_path, ml_model_path
            )
            
            logger.info(f"Scored {len(hybrid_df)} candidate pairs")
            
            self._end_stage_timer("candidate_scoring")
            return hybrid_df
            
        except Exception as e:
            logger.error(f"Candidate scoring failed: {e}")
            raise
    
    def merge_records(self, hybrid_df: pd.DataFrame, 
                     providers_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Merge records based on scoring decisions.
        
        Args:
            hybrid_df: DataFrame with hybrid scores
            providers_df: Provider records DataFrame
            
        Returns:
            Tuple of (merged_providers_df, unmerged_providers_df, merge_log_df)
        """
        self._start_stage_timer("record_merging")
        
        try:
            merge_config = self.config.get("merge", {})
            merged_df, unmerged_df, merge_log_df = merge_provider_records(
                providers_df, hybrid_df, merge_config
            )
            
            logger.info(f"Merged records: {len(merged_df)} canonical, {len(unmerged_df)} unmerged")
            
            self._end_stage_timer("record_merging")
            return merged_df, unmerged_df, merge_log_df
            
        except Exception as e:
            logger.error(f"Record merging failed: {e}")
            raise
    
    def process_audit_queue(self, hybrid_df: pd.DataFrame):
        """
        Add borderline cases to audit queue.
        
        Args:
            hybrid_df: DataFrame with hybrid scores
        """
        self._start_stage_timer("audit_processing")
        
        try:
            # Filter audit candidates (borderline scores)
            audit_config = self.config.get("audit", {})
            audit_low = self.config.get("scoring", {}).get("thresholds", {}).get("audit_low", 0.65)
            audit_high = self.config.get("scoring", {}).get("thresholds", {}).get("audit_high", 0.85)
            
            audit_candidates = hybrid_df[
                (hybrid_df["hybrid_score"] >= audit_low) & 
                (hybrid_df["hybrid_score"] < audit_high)
            ].copy()
            
            if not audit_candidates.empty:
                self.audit_logger.add_to_audit_queue(audit_candidates)
                logger.info(f"Added {len(audit_candidates)} pairs to audit queue")
            
            self._end_stage_timer("audit_processing")
            
        except Exception as e:
            logger.error(f"Audit processing failed: {e}")
            raise
    
    def generate_report(self, merge_log_df: pd.DataFrame, 
                       original_count: int) -> Dict[str, any]:
        """
        Generate pipeline performance report.
        
        Args:
            merge_log_df: Merge log DataFrame
            original_count: Original number of records
            
        Returns:
            Performance report dictionary
        """
        self._start_stage_timer("report_generation")
        
        try:
            # Calculate merge statistics
            from merge.merger import ProviderMerger
            merger = ProviderMerger(self.config.get("merge", {}))
            merge_stats = merger.get_merge_statistics(merge_log_df, original_count)
            
            # Get audit statistics
            audit_metrics = self.audit_logger.calculate_audit_metrics()
            
            # Compile report
            report = {
                "pipeline_execution": {
                    "start_time": self.pipeline_start_time,
                    "end_time": datetime.now(),
                    "stage_times": self.stage_times,
                    "total_duration": time.time() - self.pipeline_start_time if self.pipeline_start_time else 0
                },
                "data_processing": {
                    "original_records": original_count,
                    "final_records": original_count - merge_stats.get("duplicate_reduction", 0),
                    "duplicate_reduction": merge_stats.get("duplicate_reduction", 0),
                    "duplicate_reduction_percentage": merge_stats.get("duplicate_reduction_percentage", 0)
                },
                "merge_statistics": merge_stats,
                "audit_metrics": audit_metrics
            }
            
            logger.info("Pipeline report generated")
            
            self._end_stage_timer("report_generation")
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise
    
    def run_pipeline(self, input_path: str, source_label: str,
                    output_path: Optional[str] = None,
                    use_spark: bool = False) -> Dict[str, any]:
        """
        Run the complete ProviderVerify pipeline.
        
        Args:
            input_path: Path to input data
            source_label: Label for data source
            output_path: Path for output files (optional)
            use_spark: Whether to use Spark for processing
            
        Returns:
            Pipeline execution report
        """
        self.pipeline_start_time = time.time()
        logger.info(f"Starting ProviderVerify pipeline for {input_path}")
        
        try:
            # 1. Data Ingestion
            raw_df = self.ingest_data(input_path, source_label, use_spark)
            original_count = len(raw_df)
            
            # 2. Data Validation
            validated_df = self.validate_data(raw_df)
            
            # 3. Data Normalization
            normalized_df = self.normalize_data(validated_df)
            
            # 4. Candidate Generation
            providers_df, candidates_df = self.generate_candidates(normalized_df)
            
            # 5. Candidate Scoring
            hybrid_df = self.score_candidates(candidates_df, providers_df)
            
            # 6. Record Merging
            merged_df, unmerged_df, merge_log_df = self.merge_records(hybrid_df, providers_df)
            
            # 7. Audit Processing
            self.process_audit_queue(hybrid_df)
            
            # 8. Report Generation
            report = self.generate_report(merge_log_df, original_count)
            
            # 9. Save results if output path specified
            if output_path:
                self._save_results(merged_df, unmerged_df, hybrid_df, merge_log_df, output_path)
            
            total_duration = time.time() - self.pipeline_start_time
            logger.info(f"Pipeline completed successfully in {total_duration:.2f} seconds")
            
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            # Cleanup Spark session if created
            if self.spark_session:
                self.spark_session.stop()
    
    def _save_results(self, merged_df: pd.DataFrame, unmerged_df: pd.DataFrame,
                     hybrid_df: pd.DataFrame, merge_log_df: pd.DataFrame,
                     output_path: str):
        """Save pipeline results to specified path."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        merged_df.to_csv(output_dir / "merged_providers.csv", index=True)
        unmerged_df.to_csv(output_dir / "unmerged_providers.csv", index=True)
        hybrid_df.to_csv(output_dir / "scoring_results.csv", index=False)
        merge_log_df.to_csv(output_dir / "merge_log.csv", index=False)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point for ProviderVerify pipeline."""
    parser = argparse.ArgumentParser(description="ProviderVerify Entity Resolution Pipeline")
    parser.add_argument("--input", required=True, help="Input data path (local file or S3 URI)")
    parser.add_argument("--source", required=True, help="Data source label")
    parser.add_argument("--config", default="config/provider_verify.yaml", help="Configuration file path")
    parser.add_argument("--output", help="Output directory path")
    parser.add_argument("--spark", action="store_true", help="Use Spark for processing")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/provider_verify.log")
        ]
    )
    
    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # Initialize and run pipeline
        pipeline = ProviderVerifyPipeline(args.config)
        report = pipeline.run_pipeline(
            input_path=args.input,
            source_label=args.source,
            output_path=args.output,
            use_spark=args.spark
        )
        
        # Print summary
        print("\n" + "="*50)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f"Original Records: {report['data_processing']['original_records']:,}")
        print(f"Final Records: {report['data_processing']['final_records']:,}")
        print(f"Duplicate Reduction: {report['data_processing']['duplicate_reduction']:,} "
              f"({report['data_processing']['duplicate_reduction_percentage']:.1f}%)")
        print(f"Total Duration: {report['pipeline_execution']['total_duration']:.2f} seconds")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
