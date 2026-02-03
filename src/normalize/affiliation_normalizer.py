"""
Affiliation normalization for ProviderVerify.

Standardizes hospital/clinic names and affiliations using fuzzy matching
and master affiliation dictionary for consistent matching.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
from thefuzz import fuzz, process
from collections import defaultdict

logger = logging.getLogger(__name__)


class AffiliationNormalizer:
    """
    Normalizes provider affiliations (hospitals, clinics, departments).
    
    Uses fuzzy matching against a master affiliation dictionary to standardize
    variations of the same institution.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize affiliation normalizer with configuration.
        
        Args:
            config: Configuration dictionary with normalization rules
        """
        self.config = config
        self.min_similarity_threshold = config.get("min_similarity_threshold", 0.5)
        self.master_affiliation_file = config.get("master_affiliation_file", "")
        
        # Master affiliation dictionary (canonical_name -> [variants])
        self.master_affiliations = {}
        self.affiliation_variants = defaultdict(list)
        
        # Load master affiliations if file exists
        if self.master_affiliation_file:
            self._load_master_affiliations()
        else:
            self._create_default_affiliations()
        
        # Compile regex patterns
        self.punctuation_pattern = re.compile(r'[^\w\s&-]')
        self.whitespace_pattern = re.compile(r'\s+')
        self.legal_suffix_pattern = re.compile(r'\b(?:Hospital|Medical Center|Clinic|Health|System|University)\b', re.IGNORECASE)
        
        logger.info("Initialized AffiliationNormalizer")
    
    def _load_master_affiliations(self):
        """Load master affiliations from CSV file."""
        try:
            master_df = pd.read_csv(self.master_affiliation_file)
            
            for _, row in master_df.iterrows():
                canonical_name = row.get('canonical_name', '').strip()
                variant = row.get('variant', '').strip()
                
                if canonical_name and variant:
                    self.master_affiliations[variant.lower()] = canonical_name
                    self.affiliation_variants[canonical_name].append(variant)
            
            logger.info(f"Loaded {len(self.master_affiliations)} affiliation variants from {self.master_affiliation_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load master affiliations file: {e}")
            self._create_default_affiliations()
    
    def _create_default_affiliations(self):
        """Create default master affiliation dictionary for common institutions."""
        default_affiliations = {
            "johns hopkins hospital": "Johns Hopkins Hospital",
            "johns hopkins": "Johns Hopkins Hospital",
            "hopkins hospital": "Johns Hopkins Hospital",
            "mayo clinic": "Mayo Clinic",
            "mayo clinic rochester": "Mayo Clinic",
            "cleveland clinic": "Cleveland Clinic",
            "cleveland clinic foundation": "Cleveland Clinic",
            "massachusetts general hospital": "Massachusetts General Hospital",
            "mass general": "Massachusetts General Hospital",
            "mgh": "Massachusetts General Hospital",
            "ucsf medical center": "UCSF Medical Center",
            "university of california san francisco": "UCSF Medical Center",
            "mount sinai hospital": "Mount Sinai Hospital",
            "mount sinai": "Mount Sinai Hospital",
            "new york presbyterian": "New York-Presbyterian Hospital",
            "nyp": "New York-Presbyterian Hospital",
            "northwestern memorial hospital": "Northwestern Memorial Hospital",
            "northwestern": "Northwestern Memorial Hospital",
            "stanford health care": "Stanford Health Care",
            "stanford hospital": "Stanford Health Care",
            "brigham and women's hospital": "Brigham and Women's Hospital",
            "brigham": "Brigham and Women's Hospital",
            "beth israel deaconess": "Beth Israel Deaconess Medical Center",
            "bidmc": "Beth Israel Deaconess Medical Center",
            "duke university hospital": "Duke University Hospital",
            "duke hospital": "Duke University Hospital",
            "vanderbilt university medical center": "Vanderbilt University Medical Center",
            "vanderbilt": "Vanderbilt University Medical Center",
        }
        
        self.master_affiliations = default_affiliations
        
        # Create reverse mapping
        for variant, canonical in default_affiliations.items():
            self.affiliation_variants[canonical].append(variant)
        
        logger.info(f"Created default master affiliations with {len(default_affiliations)} variants")
    
    def normalize_affiliation(self, affiliation: str) -> str:
        """
        Normalize a single affiliation string.
        
        Args:
            affiliation: Raw affiliation string
            
        Returns:
            Normalized affiliation
        """
        if pd.isna(affiliation) or not isinstance(affiliation, str):
            return ""
        
        # Convert to lowercase and strip
        affiliation = str(affiliation).lower().strip()
        
        # Remove punctuation except ampersands and hyphens
        affiliation = self.punctuation_pattern.sub(' ', affiliation)
        
        # Normalize whitespace
        affiliation = self.whitespace_pattern.sub(' ', affiliation).strip()
        
        # Check against master affiliations
        if affiliation in self.master_affiliations:
            return self.master_affiliations[affiliation]
        
        # Try fuzzy matching
        best_match = self._find_best_match(affiliation)
        if best_match and best_match[1] >= self.min_similarity_threshold:
            return best_match[0]
        
        # Return cleaned affiliation if no match found
        return affiliation.title()
    
    def _find_best_match(self, affiliation: str) -> Tuple[str, float]:
        """
        Find best fuzzy match for affiliation.
        
        Args:
            affiliation: Normalized affiliation string
            
        Returns:
            Tuple of (matched_canonical_name, similarity_score)
        """
        # Get list of canonical names
        canonical_names = list(set(self.master_affiliations.values()))
        
        if not canonical_names:
            return "", 0.0
        
        # Use fuzzy matching
        best_match = process.extractOne(affiliation, canonical_names, scorer=fuzz.ratio)
        
        if best_match:
            return best_match[0], best_match[1] / 100.0
        else:
            return "", 0.0
    
    def extract_affiliation_tokens(self, affiliation: str) -> List[str]:
        """
        Extract tokens from affiliation for matching.
        
        Args:
            affiliation: Normalized affiliation
            
        Returns:
            List of affiliation tokens
        """
        if pd.isna(affiliation) or not isinstance(affiliation, str):
            return []
        
        # Split into tokens and filter out common stop words
        stop_words = {"hospital", "medical", "center", "clinic", "health", "system", "university", "of", "the", "and"}
        
        tokens = [token.strip() for token in affiliation.lower().split() 
                 if token.strip() and token not in stop_words]
        
        return tokens
    
    def calculate_affiliation_similarity(self, affiliation1: str, affiliation2: str) -> Dict[str, float]:
        """
        Calculate similarity metrics between two affiliations.
        
        Args:
            affiliation1: First normalized affiliation
            affiliation2: Second normalized affiliation
            
        Returns:
            Dictionary with similarity scores
        """
        if pd.isna(affiliation1) or pd.isna(affiliation2) or not isinstance(affiliation1, str) or not isinstance(affiliation2, str):
            return {
                "exact_match": 0.0,
                "fuzz_ratio": 0.0,
                "token_jaccard": 0.0,
                "canonical_match": 0.0
            }
        
        # Exact match
        exact_match = 1.0 if affiliation1.lower() == affiliation2.lower() else 0.0
        
        # Fuzzy ratio
        fuzz_ratio = fuzz.ratio(affiliation1, affiliation2) / 100.0
        
        # Token Jaccard similarity
        tokens1 = set(self.extract_affiliation_tokens(affiliation1))
        tokens2 = set(self.extract_affiliation_tokens(affiliation2))
        
        if tokens1 or tokens2:
            token_jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
        else:
            token_jaccard = 0.0
        
        # Canonical match (both map to same canonical name)
        canonical1 = self.normalize_affiliation(affiliation1)
        canonical2 = self.normalize_affiliation(affiliation2)
        canonical_match = 1.0 if canonical1 == canonical2 else 0.0
        
        return {
            "exact_match": exact_match,
            "fuzz_ratio": fuzz_ratio,
            "token_jaccard": token_jaccard,
            "canonical_match": canonical_match
        }
    
    def normalize_dataframe(self, df: pd.DataFrame, 
                          affiliation_column: str = "affiliation") -> pd.DataFrame:
        """
        Normalize affiliations in a DataFrame.
        
        Args:
            df: Input DataFrame
            affiliation_column: Column with affiliations
            
        Returns:
            DataFrame with normalized affiliation columns
        """
        result_df = df.copy()
        
        if affiliation_column not in df.columns:
            logger.warning(f"Affiliation column '{affiliation_column}' not found in DataFrame")
            return result_df
        
        # Normalize affiliation
        result_df[f"{affiliation_column}_norm"] = df[affiliation_column].apply(self.normalize_affiliation)
        
        # Extract affiliation tokens
        result_df["affiliation_tokens"] = result_df[f"{affiliation_column}_norm"].apply(self.extract_affiliation_tokens)
        
        logger.info(f"Normalized affiliations for {len(result_df)} records")
        return result_df
    
    def add_master_affiliation(self, canonical_name: str, variants: List[str]):
        """
        Add new affiliation to master dictionary.
        
        Args:
            canonical_name: Canonical affiliation name
            variants: List of variant spellings
        """
        canonical_name = canonical_name.strip()
        
        for variant in variants:
            variant = variant.strip().lower()
            self.master_affiliations[variant] = canonical_name
            self.affiliation_variants[canonical_name].append(variant)
        
        logger.info(f"Added {len(variants)} variants for canonical affiliation '{canonical_name}'")
    
    def get_affiliation_stats(self) -> Dict[str, int]:
        """
        Get statistics about the affiliation dictionary.
        
        Returns:
            Dictionary with affiliation statistics
        """
        return {
            "total_canonical_affiliations": len(self.affiliation_variants),
            "total_variants": len(self.master_affiliations),
            "avg_variants_per_canonical": len(self.master_affiliations) / len(self.affiliation_variants) if self.affiliation_variants else 0
        }


def normalize_provider_affiliations(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Convenience function to normalize provider affiliations in a DataFrame.
    
    Args:
        df: Provider DataFrame
        config: Normalization configuration
        
    Returns:
        DataFrame with normalized affiliations
    """
    normalizer = AffiliationNormalizer(config)
    return normalizer.normalize_dataframe(df)
