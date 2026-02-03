"""
Integration tests for the complete ProviderVerify pipeline.
"""

import pytest
import pandas as pd
import tempfile
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.run_provider_verify import ProviderVerifyPipeline


class TestProviderVerifyPipeline:
    """Integration tests for the complete pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_data = pd.DataFrame({
            "provider_id": ["1", "2", "3", "4", "5"],
            "raw_name": [
                "Dr. John Smith MD",
                "John Smith MD", 
                "Dr. Jane Doe PhD",
                "Jane Doe PhD",
                "Dr. Robert Johnson DO"
            ],
            "first_name": ["John", "John", "Jane", "Jane", "Robert"],
            "last_name": ["Smith", "Smith", "Doe", "Doe", "Johnson"],
            "affiliation": [
                "Johns Hopkins Hospital",
                "Johns Hopkins Hospital",
                "Mayo Clinic",
                "Mayo Clinic", 
                "Cleveland Clinic"
            ],
            "address": [
                "600 N Wolfe St, Baltimore, MD 21287",
                "600 N Wolfe Street, Baltimore, MD 21287",
                "200 First St SW, Rochester, MN 55905",
                "200 First Street SW, Rochester, MN 55905",
                "9500 Euclid Ave, Cleveland, OH 44195"
            ],
            "phone": [
                "(410) 955-5000",
                "410-955-5000",
                "(507) 284-2511",
                "507-284-2511",
                "(216) 444-2200"
            ],
            "email": [
                "jsmith@jhmi.edu",
                "john.smith@hopkins.edu",
                "jdoe@mayo.edu",
                "jane.doe@mayo.edu",
                "rjohnson@ccf.org"
            ],
            "source": ["EHR", "Medicaid", "EHR", "HMO", "EHR"]
        })
        
        # Save test data to temporary file
        self.test_file = Path(self.temp_dir) / "test_providers.csv"
        self.test_data.to_csv(self.test_file, index=False)
        
        # Create minimal config
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        self.create_test_config()
    
    def create_test_config(self):
        """Create minimal test configuration."""
        config_content = """
data_sources:
  bucket_name: "test-bucket"
  supported_formats: ["csv"]

schema:
  required_columns:
    - provider_id
    - raw_name
    - first_name
    - last_name
    - affiliation
    - address
    - phone
    - email
    - source

normalization:
  name:
    remove_titles: ["Dr.", "Dr", "MD", "PhD", "DO"]
    remove_suffixes: ["Jr.", "Sr.", "II", "III"]
    case_format: "title"
  affiliation:
    min_similarity_threshold: 0.5
  address:
    standardize_city_state: true
    normalize_street: true
  phone:
    default_country_code: "US"
  email:
    case_sensitive: false

blocking:
  strategies:
    - name: "location_name"
      keys: ["state_norm", "zip_norm[:3]", "last_name_metaphone", "first_name_soundex"]
  max_candidates_per_block: 1000

scoring:
  weights:
    name_exact: 0.30
    name_fuzzy: 0.20
    name_fuzzy_fuzz: 0.15
    affiliation: 0.15
    location: 0.25
    contact: 0.10
  thresholds:
    auto_merge: 0.85
    audit_low: 0.65
    audit_high: 0.85

merge:
  conflict_resolution:
    strategy: "most_complete"
    keep_all_values: true
  canonical_id:
    algorithm: "uuid4_sha256"

audit:
  sample_rate: 0.07

reporting:
  metrics:
    - precision
    - recall
    - f1_score

security:
  encryption_at_rest: true
  audit_logging: true
"""
        self.config_path.write_text(config_content)
    
    def test_full_pipeline_execution(self):
        """Test complete pipeline execution."""
        # Initialize pipeline
        pipeline = ProviderVerifyPipeline(str(self.config_path))
        
        # Run pipeline
        output_path = Path(self.temp_dir) / "output"
        report = pipeline.run_pipeline(
            input_path=str(self.test_file),
            source_label="TEST",
            output_path=str(output_path),
            use_spark=False
        )
        
        # Verify report structure
        assert "pipeline_execution" in report
        assert "data_processing" in report
        assert "merge_statistics" in report
        assert "audit_metrics" in report
        
        # Verify data processing results
        data_proc = report["data_processing"]
        assert data_proc["original_records"] == 5
        assert data_proc["final_records"] <= 5  # Should be less due to merging
        assert data_proc["duplicate_reduction"] >= 0
        
        # Verify output files exist
        assert (output_path / "merged_providers.csv").exists()
        assert (output_path / "unmerged_providers.csv").exists()
        assert (output_path / "scoring_results.csv").exists()
        assert (output_path / "merge_log.csv").exists()
        
        # Verify merged providers
        merged_df = pd.read_csv(output_path / "merged_providers.csv", index_col=0)
        assert len(merged_df) >= 1  # At least one merge should occur
        
        # Verify scoring results
        scoring_df = pd.read_csv(output_path / "scoring_results.csv")
        assert len(scoring_df) >= 1  # Should have candidate pairs
        assert "hybrid_score" in scoring_df.columns
        assert "decision" in scoring_df.columns
    
    def test_data_ingestion(self):
        """Test data ingestion component."""
        pipeline = ProviderVerifyPipeline(str(self.config_path))
        
        # Test ingestion
        df = pipeline.ingest_data(str(self.test_file), "TEST", use_spark=False)
        
        assert len(df) == 5
        assert "source" in df.columns
        assert all(df["source"] == "TEST")
        assert list(df.columns) == list(self.test_data.columns) + ["source"]
    
    def test_data_validation(self):
        """Test data validation component."""
        pipeline = ProviderVerifyPipeline(str(self.config_path))
        
        # Load and validate data
        df = pipeline.ingest_data(str(self.test_file), "TEST", use_spark=False)
        validated_df = pipeline.validate_data(df)
        
        assert len(validated_df) <= len(df)  # Validation may remove invalid records
        assert len(validated_df) > 0  # Should have some valid records
    
    def test_data_normalization(self):
        """Test data normalization component."""
        pipeline = ProviderVerifyPipeline(str(self.config_path))
        
        # Load and normalize data
        df = pipeline.ingest_data(str(self.test_file), "TEST", use_spark=False)
        validated_df = pipeline.validate_data(df)
        normalized_df = pipeline.normalize_data(validated_df)
        
        # Check for normalized columns
        assert "norm_name" in normalized_df.columns
        assert "first_name_norm" in normalized_df.columns
        assert "last_name_norm" in normalized_df.columns
        assert "norm_affiliation" in normalized_df.columns
        assert "city_norm" in normalized_df.columns
        assert "state_norm" in normalized_df.columns
        assert "zip_norm" in normalized_df.columns
        assert "phone_norm" in normalized_df.columns
        assert "email_norm" in normalized_df.columns
        
        # Check normalization results
        assert all(normalized_df["norm_name"].str.title() == normalized_df["norm_name"])
        assert all(normalized_df["state_norm"].str.len() == 2)  # State abbreviations
        assert all(normalized_df["zip_norm"].str.match(r"\d{5}"))  # 5-digit ZIPs
    
    def test_candidate_generation(self):
        """Test candidate generation component."""
        pipeline = ProviderVerifyPipeline(str(self.config_path))
        
        # Run through normalization
        df = pipeline.ingest_data(str(self.test_file), "TEST", use_spark=False)
        validated_df = pipeline.validate_data(df)
        normalized_df = pipeline.normalize_data(validated_df)
        
        # Generate candidates
        providers_df, candidates_df = pipeline.generate_candidates(normalized_df)
        
        assert len(candidates_df) > 0  # Should generate some candidates
        assert "record_id_1" in candidates_df.columns
        assert "record_id_2" in candidates_df.columns
        assert "block_key" in candidates_df.columns
        assert "strategy" in candidates_df.columns
        
        # Check block keys in providers
        block_columns = [col for col in providers_df.columns if col.startswith("block_")]
        assert len(block_columns) > 0
    
    def test_candidate_scoring(self):
        """Test candidate scoring component."""
        pipeline = ProviderVerifyPipeline(str(self.config_path))
        
        # Run through candidate generation
        df = pipeline.ingest_data(str(self.test_file), "TEST", use_spark=False)
        validated_df = pipeline.validate_data(df)
        normalized_df = pipeline.normalize_data(validated_df)
        providers_df, candidates_df = pipeline.generate_candidates(normalized_df)
        
        # Score candidates
        hybrid_df = pipeline.score_candidates(candidates_df, providers_df)
        
        assert len(hybrid_df) == len(candidates_df)
        assert "hybrid_score" in hybrid_df.columns
        assert "decision" in hybrid_df.columns
        assert "auto_merge" in hybrid_df.columns
        assert "audit_required" in hybrid_df.columns
        
        # Check score ranges
        assert all(0 <= hybrid_df["hybrid_score"] <= 1)
        assert all(hybrid_df["decision"].isin(["AUTO_MERGE", "AUDIT", "REJECT"]))
    
    def test_record_merging(self):
        """Test record merging component."""
        pipeline = ProviderVerifyPipeline(str(self.config_path))
        
        # Run through scoring
        df = pipeline.ingest_data(str(self.test_file), "TEST", use_spark=False)
        validated_df = pipeline.validate_data(df)
        normalized_df = pipeline.normalize_data(validated_df)
        providers_df, candidates_df = pipeline.generate_candidates(normalized_df)
        hybrid_df = pipeline.score_candidates(candidates_df, providers_df)
        
        # Merge records
        merged_df, unmerged_df, merge_log_df = pipeline.merge_records(hybrid_df, providers_df)
        
        assert len(merged_df) + len(unmerged_df) == len(providers_df)
        assert "canonical_id" in merged_df.columns
        assert "merged_from" in merged_df.columns
        assert "merge_count" in merged_df.columns
        
        # Check merge log
        assert len(merge_log_df) == len(merged_df)
        assert "canonical_id" in merge_log_df.columns
        assert "merged_record_ids" in merge_log_df.columns
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__])
