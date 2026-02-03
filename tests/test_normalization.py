"""
Unit tests for normalization modules.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.normalize.name_normalizer import NameNormalizer
from src.normalize.affiliation_normalizer import AffiliationNormalizer
from src.normalize.address_normalizer import AddressNormalizer


class TestNameNormalizer:
    """Test cases for name normalization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "remove_titles": ["Dr.", "Dr", "MD", "PhD"],
            "remove_suffixes": ["Jr.", "Sr.", "II", "III"],
            "case_format": "title"
        }
        self.normalizer = NameNormalizer(self.config)
    
    def test_normalize_name_basic(self):
        """Test basic name normalization."""
        assert self.normalizer.normalize_name("Dr. John Smith MD") == "John Smith"
        assert self.normalizer.normalize_name("  john  smith  ") == "John Smith"
        assert self.normalizer.normalize_name("") == ""
        assert self.normalizer.normalize_name(None) == ""
    
    def test_split_name_components(self):
        """Test name component splitting."""
        first, last, middle = self.normalizer.split_name_components("John Smith")
        assert first == "John"
        assert last == "Smith"
        assert middle == ""
        
        first, last, middle = self.normalizer.split_name_components("John A Smith Jr")
        assert first == "John"
        assert last == "Jr"
        assert middle == "A Smith"
    
    def test_phonetic_encodings(self):
        """Test phonetic encoding generation."""
        encodings = self.normalizer.get_phonetic_encodings("John Smith")
        assert "metaphone" in encodings
        assert "soundex" in encodings
        assert encodings["metaphone"] != ""
        assert encodings["soundex"] != ""
    
    def test_name_similarity(self):
        """Test name similarity calculation."""
        sim = self.normalizer.calculate_name_similarity("John Smith", "John Smith")
        assert sim["exact_match"] == 1.0
        assert sim["fuzz_ratio"] == 1.0
        
        sim = self.normalizer.calculate_name_similarity("John Smith", "Jane Doe")
        assert sim["exact_match"] == 0.0
        assert sim["fuzz_ratio"] < 1.0


class TestAffiliationNormalizer:
    """Test cases for affiliation normalization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "min_similarity_threshold": 0.5,
            "master_affiliation_file": ""
        }
        self.normalizer = AffiliationNormalizer(self.config)
    
    def test_normalize_affiliation_basic(self):
        """Test basic affiliation normalization."""
        assert self.normalizer.normalize_affiliation("Johns Hopkins Hospital") == "Johns Hopkins Hospital"
        assert self.normalizer.normalize_affiliation("  mayo clinic  ") == "Mayo Clinic"
        assert self.normalizer.normalize_affiliation("") == ""
        assert self.normalizer.normalize_affiliation(None) == ""
    
    def test_affiliation_similarity(self):
        """Test affiliation similarity calculation."""
        sim = self.normalizer.calculate_affiliation_similarity(
            "Johns Hopkins Hospital", "Johns Hopkins Hospital"
        )
        assert sim["exact_match"] == 1.0
        assert sim["jaccard_similarity"] == 1.0
        
        sim = self.normalizer.calculate_affiliation_similarity(
            "Johns Hopkins Hospital", "Mayo Clinic"
        )
        assert sim["exact_match"] == 0.0
        assert sim["jaccard_similarity"] < 1.0


class TestAddressNormalizer:
    """Test cases for address normalization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "standardize_city_state": True,
            "normalize_street": True,
            "zip_regex": "\\d{5}(-\\d{4})?"
        }
        self.normalizer = AddressNormalizer(self.config)
    
    def test_parse_address(self):
        """Test address parsing."""
        components = self.normalizer.parse_address("600 N Wolfe St, Baltimore, MD 21287")
        assert components["city"] == "Baltimore"
        assert components["state"] == "MD"
        assert components["zipcode"] == "21287"
        assert components["street_number"] == "600"
    
    def test_normalize_city(self):
        """Test city normalization."""
        assert self.normalizer.normalize_city("baltimore") == "Baltimore"
        assert self.normalizer.normalize_city("  new york  ") == "New York"
        assert self.normalizer.normalize_city("") == ""
    
    def test_normalize_state(self):
        """Test state normalization."""
        assert self.normalizer.normalize_state("Maryland") == "MD"
        assert self.normalizer.normalize_state("md") == "MD"
        assert self.normalizer.normalize_state("California") == "CA"
        assert self.normalizer.normalize_state("") == ""
    
    def test_normalize_zipcode(self):
        """Test ZIP code normalization."""
        assert self.normalizer.normalize_zipcode("21287-1234") == "21287"
        assert self.normalizer.normalize_zipcode("21287") == "21287"
        assert self.normalizer.normalize_zipcode("") == ""
    
    def test_normalize_phone(self):
        """Test phone normalization."""
        phone = self.normalizer.normalize_phone("(410) 955-5000")
        assert phone.startswith("+")  # Should be in E164 format
        
        assert self.normalizer.normalize_phone("") == ""
        assert self.normalizer.normalize_phone("invalid") == ""
    
    def test_normalize_email(self):
        """Test email normalization."""
        assert self.normalizer.normalize_email("John.Smith@Example.COM") == "john.smith@example.com"
        assert self.normalizer.normalize_email("") == ""
        assert self.normalizer.normalize_email("invalid-email") == ""


if __name__ == "__main__":
    pytest.main([__file__])
