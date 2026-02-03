"""
Deterministic matching rules for ProviderVerify.

Implements rule-based similarity scoring for provider records
including exact matches, fuzzy matches, and field-specific rules.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
from thefuzz import fuzz
from Levenshtein import distance as levenshtein_distance
from jellyfish import jaro_winkler_similarity
import numpy as np

logger = logging.getLogger(__name__)


class DeterministicMatcher:
    """
    Implements deterministic matching rules for provider similarity scoring.
    
    Combines exact matches, fuzzy string matching, and field-specific rules
    to calculate similarity scores between provider records.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize deterministic matcher with configuration.
        
        Args:
            config: Configuration dictionary with scoring weights and thresholds
        """
        self.config = config
        self.weights = config.get("weights", {})
        self.fuzzy_thresholds = config.get("fuzzy_thresholds", {})
        
        # Set default weights if not provided
        self.default_weights = {
            "name_exact": 0.30,
            "name_fuzzy": 0.20,
            "name_fuzzy_fuzz": 0.15,
            "affiliation": 0.15,
            "location": 0.25,
            "contact": 0.10
        }
        
        # Merge with provided weights
        for key, value in self.default_weights.items():
            if key not in self.weights:
                self.weights[key] = value
        
        # Set default fuzzy thresholds
        self.default_fuzzy_thresholds = {
            "levenshtein_max_distance": 2,
            "fuzz_ratio_min": 90,
            "jaccard_min_similarity": 0.3
        }
        
        for key, value in self.default_fuzzy_thresholds.items():
            if key not in self.fuzzy_thresholds:
                self.fuzzy_thresholds[key] = value
        
        logger.info("Initialized DeterministicMatcher")
    
    def calculate_name_similarity(self, name1: str, name2: str) -> Dict[str, float]:
        """
        Calculate name similarity using multiple methods.
        
        Args:
            name1: First normalized name
            name2: Second normalized name
            
        Returns:
            Dictionary with name similarity scores
        """
        if not name1 or not name2:
            return {
                "exact_match": 0.0,
                "levenshtein_match": 0.0,
                "fuzz_ratio": 0.0,
                "jaro_winkler": 0.0
            }
        
        # Exact match
        exact_match = 1.0 if name1.lower() == name2.lower() else 0.0
        
        # Levenshtein distance match
        max_distance = self.fuzzy_thresholds.get("levenshtein_max_distance", 2)
        levenshtein_dist = levenshtein_distance(name1.lower(), name2.lower())
        levenshtein_match = 1.0 if levenshtein_dist <= max_distance else 0.0
        
        # Fuzzy ratio
        fuzz_ratio_score = fuzz.ratio(name1, name2) / 100.0
        fuzz_ratio_match = 1.0 if fuzz_ratio_score >= self.fuzzy_thresholds.get("fuzz_ratio_min", 90) / 100.0 else 0.0
        
        # Jaro-Winkler similarity
        jaro_winkler_score = jaro_winkler_similarity(name1.lower(), name2.lower())
        
        return {
            "exact_match": exact_match,
            "levenshtein_match": levenshtein_match,
            "fuzz_ratio": fuzz_ratio_score,
            "fuzz_ratio_match": fuzz_ratio_match,
            "jaro_winkler": jaro_winkler_score
        }
    
    def calculate_affiliation_similarity(self, aff1: str, aff2: str) -> Dict[str, float]:
        """
        Calculate affiliation similarity using Jaccard similarity.
        
        Args:
            aff1: First normalized affiliation
            aff2: Second normalized affiliation
            
        Returns:
            Dictionary with affiliation similarity scores
        """
        if not aff1 or not aff2:
            return {
                "exact_match": 0.0,
                "jaccard_similarity": 0.0,
                "fuzz_ratio": 0.0
            }
        
        # Exact match
        exact_match = 1.0 if aff1.lower() == aff2.lower() else 0.0
        
        # Token-based Jaccard similarity
        tokens1 = set(aff1.lower().split())
        tokens2 = set(aff2.lower().split())
        
        if tokens1 or tokens2:
            jaccard_sim = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
        else:
            jaccard_sim = 0.0
        
        # Fuzzy ratio
        fuzz_ratio_score = fuzz.ratio(aff1, aff2) / 100.0
        
        return {
            "exact_match": exact_match,
            "jaccard_similarity": jaccard_sim,
            "fuzz_ratio": fuzz_ratio_score
        }
    
    def calculate_location_similarity(self, addr1: Dict, addr2: Dict) -> Dict[str, float]:
        """
        Calculate location similarity based on address components.
        
        Args:
            addr1: First address components dictionary
            addr2: Second address components dictionary
            
        Returns:
            Dictionary with location similarity scores
        """
        similarities = {
            "city_match": 0.0,
            "state_match": 0.0,
            "zip_match": 0.0,
            "street_match": 0.0,
            "location_match": 0.0
        }
        
        # City match
        city1 = addr1.get("city_norm", "").lower()
        city2 = addr2.get("city_norm", "").lower()
        similarities["city_match"] = 1.0 if city1 == city2 and city1 != "" else 0.0
        
        # State match
        state1 = addr1.get("state_norm", "").upper()
        state2 = addr2.get("state_norm", "").upper()
        similarities["state_match"] = 1.0 if state1 == state2 and state1 != "" else 0.0
        
        # ZIP match
        zip1 = addr1.get("zip_norm", "")
        zip2 = addr2.get("zip_norm", "")
        similarities["zip_match"] = 1.0 if zip1 == zip2 and zip1 != "" else 0.0
        
        # Street match
        street1 = addr1.get("address_line_1_norm", "").lower()
        street2 = addr2.get("address_line_1_norm", "").lower()
        similarities["street_match"] = 1.0 if street1 == street2 and street1 != "" else 0.0
        
        # Overall location match (city+state or ZIP)
        location_match = 0.0
        if (similarities["city_match"] == 1.0 and similarities["state_match"] == 1.0):
            location_match = 1.0
        elif similarities["zip_match"] == 1.0:
            location_match = 1.0
        
        similarities["location_match"] = location_match
        
        return similarities
    
    def calculate_contact_similarity(self, contact1: Dict, contact2: Dict) -> Dict[str, float]:
        """
        Calculate contact similarity (phone/email).
        
        Args:
            contact1: First contact info dictionary
            contact2: Second contact info dictionary
            
        Returns:
            Dictionary with contact similarity scores
        """
        similarities = {
            "phone_match": 0.0,
            "email_match": 0.0,
            "contact_match": 0.0
        }
        
        # Phone match
        phone1 = contact1.get("phone_norm", "")
        phone2 = contact2.get("phone_norm", "")
        similarities["phone_match"] = 1.0 if phone1 == phone2 and phone1 != "" else 0.0
        
        # Email match
        email1 = contact1.get("email_norm", "").lower()
        email2 = contact2.get("email_norm", "").lower()
        similarities["email_match"] = 1.0 if email1 == email2 and email1 != "" else 0.0
        
        # Overall contact match
        similarities["contact_match"] = 1.0 if (similarities["phone_match"] == 1.0 or 
                                               similarities["email_match"] == 1.0) else 0.0
        
        return similarities
    
    def calculate_deterministic_score(self, rec1: Dict, rec2: Dict) -> Dict[str, float]:
        """
        Calculate comprehensive deterministic similarity score between two records.
        
        Args:
            rec1: First provider record
            rec2: Second provider record
            
        Returns:
            Dictionary with detailed similarity scores and total score
        """
        scores = {}
        
        # Name similarity
        name1 = rec1.get("norm_name", "")
        name2 = rec2.get("norm_name", "")
        name_sim = self.calculate_name_similarity(name1, name2)
        scores.update({f"name_{k}": v for k, v in name_sim.items()})
        
        # Affiliation similarity
        aff1 = rec1.get("norm_affiliation", "")
        aff2 = rec2.get("norm_affiliation", "")
        aff_sim = self.calculate_affiliation_similarity(aff1, aff2)
        scores.update({f"affiliation_{k}": v for k, v in aff_sim.items()})
        
        # Location similarity
        addr1 = {
            "city_norm": rec1.get("city_norm", ""),
            "state_norm": rec1.get("state_norm", ""),
            "zip_norm": rec1.get("zip_norm", ""),
            "address_line_1_norm": rec1.get("address_line_1_norm", "")
        }
        addr2 = {
            "city_norm": rec2.get("city_norm", ""),
            "state_norm": rec2.get("state_norm", ""),
            "zip_norm": rec2.get("zip_norm", ""),
            "address_line_1_norm": rec2.get("address_line_1_norm", "")
        }
        loc_sim = self.calculate_location_similarity(addr1, addr2)
        scores.update({f"location_{k}": v for k, v in loc_sim.items()})
        
        # Contact similarity
        contact1 = {
            "phone_norm": rec1.get("phone_norm", ""),
            "email_norm": rec1.get("email_norm", "")
        }
        contact2 = {
            "phone_norm": rec2.get("phone_norm", ""),
            "email_norm": rec2.get("email_norm", "")
        }
        contact_sim = self.calculate_contact_similarity(contact1, contact2)
        scores.update({f"contact_{k}": v for k, v in contact_sim.items()})
        
        # Calculate weighted total score
        total_score = (
            self.weights.get("name_exact", 0.30) * scores.get("name_exact_match", 0.0) +
            self.weights.get("name_fuzzy", 0.20) * scores.get("name_levenshtein_match", 0.0) +
            self.weights.get("name_fuzzy_fuzz", 0.15) * scores.get("name_fuzz_ratio_match", 0.0) +
            self.weights.get("affiliation", 0.15) * scores.get("affiliation_jaccard_similarity", 0.0) +
            self.weights.get("location", 0.25) * scores.get("location_location_match", 0.0) +
            self.weights.get("contact", 0.10) * scores.get("contact_contact_match", 0.0)
        )
        
        scores["deterministic_score"] = total_score
        
        return scores
    
    def score_candidate_pairs(self, candidates_df: pd.DataFrame, 
                           providers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all candidate pairs using deterministic rules.
        
        Args:
            candidates_df: DataFrame with candidate pairs
            providers_df: DataFrame with provider records
            
        Returns:
            DataFrame with similarity scores for each pair
        """
        scored_pairs = []
        
        for _, pair in candidates_df.iterrows():
            id1 = pair["record_id_1"]
            id2 = pair["record_id_2"]
            
            # Get provider records
            rec1 = providers_df.loc[id1].to_dict()
            rec2 = providers_df.loc[id2].to_dict()
            
            # Calculate similarity scores
            scores = self.calculate_deterministic_score(rec1, rec2)
            
            # Add pair information
            pair_data = {
                "record_id_1": id1,
                "record_id_2": id2,
                "block_key": pair.get("block_key", ""),
                "strategy": pair.get("strategy", ""),
                **scores
            }
            
            scored_pairs.append(pair_data)
        
        scored_df = pd.DataFrame(scored_pairs)
        
        logger.info(f"Scored {len(scored_df)} candidate pairs using deterministic rules")
        return scored_df
    
    def get_score_statistics(self, scored_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistics for deterministic scores.
        
        Args:
            scored_df: DataFrame with scored pairs
            
        Returns:
            Dictionary with score statistics
        """
        if "deterministic_score" not in scored_df.columns:
            return {}
        
        scores = scored_df["deterministic_score"]
        
        stats = {
            "mean_score": scores.mean(),
            "median_score": scores.median(),
            "std_score": scores.std(),
            "min_score": scores.min(),
            "max_score": scores.max(),
            "score_distribution": {
                "0.0-0.2": (scores <= 0.2).sum(),
                "0.2-0.4": ((scores > 0.2) & (scores <= 0.4)).sum(),
                "0.4-0.6": ((scores > 0.4) & (scores <= 0.6)).sum(),
                "0.6-0.8": ((scores > 0.6) & (scores <= 0.8)).sum(),
                "0.8-1.0": (scores > 0.8).sum()
            }
        }
        
        return stats


def apply_deterministic_scoring(candidates_df: pd.DataFrame, 
                               providers_df: pd.DataFrame, 
                               config: Dict) -> pd.DataFrame:
    """
    Convenience function to apply deterministic scoring to candidate pairs.
    
    Args:
        candidates_df: Candidate pairs DataFrame
        providers_df: Provider records DataFrame
        config: Scoring configuration
        
    Returns:
        DataFrame with deterministic scores
    """
    matcher = DeterministicMatcher(config)
    scored_df = matcher.score_candidate_pairs(candidates_df, providers_df)
    
    # Get statistics
    stats = matcher.get_score_statistics(scored_df)
    logger.info(f"Deterministic scoring completed: mean score = {stats.get('mean_score', 0):.3f}")
    
    return scored_df
