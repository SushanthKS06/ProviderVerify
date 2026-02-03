"""
Unit tests for blocking module.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.blocking.block_key_builder import BlockKeyBuilder


class TestBlockKeyBuilder:
    """Test cases for block key builder."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "strategies": [
                {
                    "name": "location_name",
                    "keys": ["state_norm", "zip_norm[:3]", "last_name_metaphone", "first_name_soundex"]
                },
                {
                    "name": "affiliation_tokens",
                    "keys": ["state_norm", "affiliation_tokens"],
                    "similarity_threshold": 0.5
                }
            ],
            "max_candidates_per_block": 1000
        }
        self.builder = BlockKeyBuilder(self.config)
    
    def test_generate_block_keys(self):
        """Test block key generation."""
        # Create test data
        df = pd.DataFrame({
            "provider_id": [1, 2, 3],
            "norm_name": ["John Smith", "Jane Doe", "Bob Johnson"],
            "first_name_norm": ["John", "Jane", "Bob"],
            "last_name_norm": ["Smith", "Doe", "Johnson"],
            "state_norm": ["MD", "MD", "NY"],
            "zip_norm": ["21287", "21287", "10001"],
            "affiliation_tokens": [["hospital", "johns", "hopkins"], ["clinic", "mayo"], ["medical", "center"]]
        })
        
        # Add phonetic encodings (simplified)
        df["last_name_metaphone"] = ["SM0", "T", "JNSN"]
        df["first_name_soundex"] = ["J500", "J000", "B100"]
        
        result_df = self.builder.generate_block_keys(df)
        
        # Check that block key columns were added
        assert "block_location_name" in result_df.columns
        assert "block_affiliation_tokens" in result_df.columns
        
        # Check that block keys are non-empty for valid records
        assert all(result_df["block_location_name"] != "")
    
    def test_generate_candidate_pairs(self):
        """Test candidate pair generation."""
        # Create test data with block keys
        df = pd.DataFrame({
            "provider_id": [1, 2, 3, 4],
            "block_location_name": ["ABC123", "ABC123", "DEF456", "ABC123"],
            "block_affiliation_tokens": ["XYZ789", "XYZ789", "XYZ789", "XYZ789"]
        })
        
        # Create candidate pairs manually
        candidates_df = pd.DataFrame({
            "record_id_1": [0, 0, 1, 2],
            "record_id_2": [1, 3, 3, 3],
            "block_key": ["ABC123", "ABC123", "ABC123", "XYZ789"],
            "strategy": ["location_name", "location_name", "location_name", "affiliation_tokens"]
        })
        
        result_df = self.builder.generate_candidate_pairs(df)
        
        # Check that we have candidate pairs
        assert len(result_df) > 0
        assert "record_id_1" in result_df.columns
        assert "record_id_2" in result_df.columns
        assert "block_key" in result_df.columns
        assert "strategy" in result_df.columns
        
        # Check that pairs are unique
        pair_keys = result_df.apply(
            lambda row: tuple(sorted([row["record_id_1"], row["record_id_2"]])), axis=1
        )
        assert len(pair_keys) == len(pair_keys.unique())
    
    def test_extract_field_value(self):
        """Test field value extraction with expressions."""
        # Test basic field reference
        row = pd.Series({"zip_norm": "21287-1234"})
        value = self.builder._extract_field_value(row, "zip_norm")
        assert value == "21287-1234"
        
        # Test slicing expression
        value = self.builder._extract_field_value(row, "zip_norm[:3]")
        assert value == "212"
        
        # Test missing field
        value = self.builder._extract_field_value(row, "missing_field")
        assert value == ""
    
    def test_extract_tokens(self):
        """Test token extraction."""
        tokens = self.builder._extract_tokens("Johns Hopkins Hospital")
        assert "johns" in tokens
        assert "hopkins" in tokens
        assert "hospital" in tokens
        
        tokens = self.builder._extract_tokens("")
        assert len(tokens) == 0
    
    def test_get_blocking_statistics(self):
        """Test blocking statistics calculation."""
        # Create test data
        df = pd.DataFrame({
            "block_location_name": ["A", "A", "B", "A", "B", "B"],
            "block_affiliation_tokens": ["X", "X", "X", "Y", "Y", "Y"]
        })
        
        candidates_df = pd.DataFrame({
            "record_id_1": [0, 1, 2],
            "record_id_2": [1, 3, 4],
            "block_key": ["A", "A", "B"],
            "strategy": ["location_name", "location_name", "affiliation_tokens"]
        })
        
        stats = self.builder.get_blocking_statistics(df, candidates_df)
        
        assert "total_records" in stats
        assert "generated_candidates" in stats
        assert "reduction_ratio" in stats
        assert "blocking_strategies" in stats
        
        assert stats["total_records"] == 6
        assert stats["generated_candidates"] == 3


if __name__ == "__main__":
    pytest.main([__file__])
