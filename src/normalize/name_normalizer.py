"""
Name normalization for ProviderVerify.

Standardizes provider names by removing titles/suffixes, normalizing case,
and applying phonetic encoding for matching.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import spacy
from jellyfish import metaphone, soundex
from thefuzz import fuzz

logger = logging.getLogger(__name__)


class NameNormalizer:
    """
    Normalizes provider names for entity resolution.
    
    Handles title/suffix removal, case normalization, and phonetic encoding.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize name normalizer with configuration.
        
        Args:
            config: Configuration dictionary with normalization rules
        """
        self.config = config
        self.remove_titles = config.get("remove_titles", ["Dr.", "Dr", "MD", "PhD", "DO", "PA", "NP", "RN"])
        self.remove_suffixes = config.get("remove_suffixes", ["Jr.", "Sr.", "II", "III", "IV"])
        self.case_format = config.get("case_format", "title")
        
        # Load spaCy model for name processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model")
        except OSError:
            logger.warning("spaCy English model not found. Using basic processing.")
            self.nlp = None
        
        # Compile regex patterns for efficiency
        self.title_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, self.remove_titles)) + r')\b\.?', re.IGNORECASE)
        self.suffix_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, self.remove_suffixes)) + r')\.?', re.IGNORECASE)
        self.punctuation_pattern = re.compile(r'[^\w\s-]')
        self.whitespace_pattern = re.compile(r'\s+')
        
        logger.info("Initialized NameNormalizer")
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize a single provider name.
        
        Args:
            name: Raw provider name
            
        Returns:
            Normalized name
        """
        if pd.isna(name) or not isinstance(name, str):
            return ""
        
        # Convert to string and strip whitespace
        name = str(name).strip()
        
        # Remove titles and suffixes
        name = self.title_pattern.sub('', name)
        name = self.suffix_pattern.sub('', name)
        
        # Remove punctuation except hyphens
        name = self.punctuation_pattern.sub(' ', name)
        
        # Normalize whitespace
        name = self.whitespace_pattern.sub(' ', name).strip()
        
        # Apply case formatting
        if self.case_format == "title":
            name = name.title()
        elif self.case_format == "upper":
            name = name.upper()
        elif self.case_format == "lower":
            name = name.lower()
        
        return name
    
    def split_name_components(self, name: str) -> Tuple[str, str, str]:
        """
        Split normalized name into first, last, and middle components.
        
        Args:
            name: Normalized provider name
            
        Returns:
            Tuple of (first_name, last_name, middle_name)
        """
        if pd.isna(name) or not isinstance(name, str):
            return "", "", ""
        
        name = name.strip()
        parts = name.split()
        
        if len(parts) == 1:
            # Single name, treat as last name
            return "", parts[0], ""
        elif len(parts) == 2:
            # First and last name
            return parts[0], parts[1], ""
        else:
            # First, middle, last name
            return parts[0], parts[-1], ' '.join(parts[1:-1])
    
    def get_phonetic_encodings(self, name: str) -> Dict[str, str]:
        """
        Generate phonetic encodings for name matching.
        
        Args:
            name: Normalized name
            
        Returns:
            Dictionary with phonetic encodings
        """
        if pd.isna(name) or not isinstance(name, str):
            return {"metaphone": "", "soundex": "", "double_metaphone": ""}
        
        # Generate phonetic encodings
        metaphone_code = metaphone(name)
        soundex_code = soundex(name)
        
        return {
            "metaphone": metaphone_code,
            "soundex": soundex_code,
            "double_metaphone": metaphone_code  # Using metaphone as double_metaphone for simplicity
        }
    
    def extract_name_tokens(self, name: str) -> List[str]:
        """
        Extract individual tokens from name for matching.
        
        Args:
            name: Normalized name
            
        Returns:
            List of name tokens
        """
        if pd.isna(name) or not isinstance(name, str):
            return []
        
        # Split into tokens and filter out empty strings
        tokens = [token.strip() for token in name.split() if token.strip()]
        
        # Use spaCy for better tokenization if available
        if self.nlp:
            doc = self.nlp(name)
            tokens = [token.text for token in doc if not token.is_space]
        
        return tokens
    
    def calculate_name_similarity(self, name1: str, name2: str) -> Dict[str, float]:
        """
        Calculate various similarity metrics between two names.
        
        Args:
            name1: First normalized name
            name2: Second normalized name
            
        Returns:
            Dictionary with similarity scores
        """
        if pd.isna(name1) or pd.isna(name2) or not isinstance(name1, str) or not isinstance(name2, str):
            return {
                "exact_match": 0.0,
                "fuzz_ratio": 0.0,
                "token_overlap": 0.0,
                "phonetic_match": 0.0
            }
        
        # Exact match
        exact_match = 1.0 if name1.lower() == name2.lower() else 0.0
        
        # Fuzzy ratio
        fuzz_ratio = fuzz.ratio(name1, name2) / 100.0
        
        # Token overlap (Jaccard similarity)
        tokens1 = set(self.extract_name_tokens(name1))
        tokens2 = set(self.extract_name_tokens(name2))
        
        if tokens1 or tokens2:
            token_overlap = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
        else:
            token_overlap = 0.0
        
        # Phonetic match
        phonetic1 = self.get_phonetic_encodings(name1)
        phonetic2 = self.get_phonetic_encodings(name2)
        
        phonetic_matches = 0
        total_phonetic = 0
        for key in ["metaphone", "soundex"]:
            if phonetic1[key] and phonetic2[key]:
                total_phonetic += 1
                if phonetic1[key] == phonetic2[key]:
                    phonetic_matches += 1
        
        phonetic_match = phonetic_matches / total_phonetic if total_phonetic > 0 else 0.0
        
        return {
            "exact_match": exact_match,
            "fuzz_ratio": fuzz_ratio,
            "token_overlap": token_overlap,
            "phonetic_match": phonetic_match
        }
    
    def normalize_dataframe(self, df: pd.DataFrame, 
                          name_column: str = "raw_name",
                          first_name_column: str = "first_name",
                          last_name_column: str = "last_name") -> pd.DataFrame:
        """
        Normalize names in a DataFrame.
        
        Args:
            df: Input DataFrame
            name_column: Column with full names
            first_name_column: Column with first names
            last_name_column: Column with last names
            
        Returns:
            DataFrame with normalized name columns
        """
        result_df = df.copy()
        
        # Normalize full name
        if name_column in df.columns:
            result_df[f"{name_column}_norm"] = df[name_column].apply(self.normalize_name)
        
        # Normalize first and last names if available
        if first_name_column in df.columns:
            result_df[f"{first_name_column}_norm"] = df[first_name_column].apply(self.normalize_name)
        
        if last_name_column in df.columns:
            result_df[f"{last_name_column}_norm"] = df[last_name_column].apply(self.normalize_name)
        
        # Generate phonetic encodings for last name
        if f"{last_name_column}_norm" in result_df.columns:
            phonetic_encodings = result_df[f"{last_name_column}_norm"].apply(self.get_phonetic_encodings)
            phonetic_df = pd.DataFrame(phonetic_encodings.tolist())
            result_df["last_name_metaphone"] = phonetic_df["metaphone"]
            result_df["last_name_soundex"] = phonetic_df["soundex"]
        
        # Generate phonetic encodings for first name
        if f"{first_name_column}_norm" in result_df.columns:
            phonetic_encodings = result_df[f"{first_name_column}_norm"].apply(self.get_phonetic_encodings)
            phonetic_df = pd.DataFrame(phonetic_encodings.tolist())
            result_df["first_name_metaphone"] = phonetic_df["metaphone"]
            result_df["first_name_soundex"] = phonetic_df["soundex"]
        
        # Extract name tokens for matching
        if f"{name_column}_norm" in result_df.columns:
            result_df["name_tokens"] = result_df[f"{name_column}_norm"].apply(self.extract_name_tokens)
        
        logger.info(f"Normalized names for {len(result_df)} records")
        return result_df


def normalize_provider_names(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Convenience function to normalize provider names in a DataFrame.
    
    Args:
        df: Provider DataFrame
        config: Normalization configuration
        
    Returns:
        DataFrame with normalized names
    """
    normalizer = NameNormalizer(config)
    return normalizer.normalize_dataframe(df)
