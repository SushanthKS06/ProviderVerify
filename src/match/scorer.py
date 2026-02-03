"""
Hybrid similarity scorer for ProviderVerify.

Combines deterministic rule-based scoring with machine learning predictions
to create a comprehensive similarity score for provider matching.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class HybridScorer:
    """
    Hybrid similarity scorer combining deterministic rules with ML predictions.
    
    Weights deterministic scores and ML probabilities to create final similarity
    scores with configurable thresholds for auto-merge and audit decisions.
    """
    
    def __init__(self, config: Dict, ml_model_path: Optional[str] = None):
        """
        Initialize hybrid scorer with configuration.
        
        Args:
            config: Configuration dictionary
            ml_model_path: Path to trained ML model (optional)
        """
        self.config = config
        self.weights = config.get("weights", {})
        self.thresholds = config.get("thresholds", {})
        
        # Set default weights
        self.default_weights = {
            "name_exact": 0.30,
            "name_fuzzy": 0.20,
            "name_fuzzy_fuzz": 0.15,
            "affiliation": 0.15,
            "location": 0.25,
            "contact": 0.10,
            "ml_model": 0.25
        }
        
        # Merge with provided weights
        for key, value in self.default_weights.items():
            if key not in self.weights:
                self.weights[key] = value
        
        # Set default thresholds
        self.default_thresholds = {
            "auto_merge": 0.85,
            "audit_low": 0.65,
            "audit_high": 0.85
        }
        
        for key, value in self.default_thresholds.items():
            if key not in self.thresholds:
                self.thresholds[key] = value
        
        # Load ML model if provided
        self.ml_predictor = None
        if ml_model_path:
            try:
                from .ml_model.predict import ProviderMatchPredictor
                self.ml_predictor = ProviderMatchPredictor(ml_model_path)
                logger.info("Loaded ML model for hybrid scoring")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")
        
        logger.info("Initialized HybridScorer")
    
    def calculate_hybrid_score(self, deterministic_scores: Dict, 
                              ml_probability: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate hybrid similarity score combining deterministic and ML scores.
        
        Args:
            deterministic_scores: Dictionary with deterministic similarity scores
            ml_probability: ML model probability (optional)
            
        Returns:
            Dictionary with hybrid score and decision
        """
        # Calculate deterministic component
        deterministic_score = (
            self.weights.get("name_exact", 0.30) * deterministic_scores.get("name_exact_match", 0.0) +
            self.weights.get("name_fuzzy", 0.20) * deterministic_scores.get("name_levenshtein_match", 0.0) +
            self.weights.get("name_fuzzy_fuzz", 0.15) * deterministic_scores.get("name_fuzz_ratio_match", 0.0) +
            self.weights.get("affiliation", 0.15) * deterministic_scores.get("affiliation_jaccard_similarity", 0.0) +
            self.weights.get("location", 0.25) * deterministic_scores.get("location_location_match", 0.0) +
            self.weights.get("contact", 0.10) * deterministic_scores.get("contact_contact_match", 0.0)
        )
        
        # Calculate hybrid score
        if ml_probability is not None and self.ml_predictor:
            # Combine deterministic and ML scores
            ml_weight = self.weights.get("ml_model", 0.25)
            deterministic_weight = 1.0 - ml_weight
            
            hybrid_score = (deterministic_weight * deterministic_score + 
                           ml_weight * ml_probability)
        else:
            # Use only deterministic score
            hybrid_score = deterministic_score
            ml_probability = 0.0
        
        # Determine decision
        decision = self._determine_decision(hybrid_score)
        
        return {
            "deterministic_score": deterministic_score,
            "ml_probability": ml_probability,
            "hybrid_score": hybrid_score,
            "decision": decision,
            "auto_merge": decision == "AUTO_MERGE",
            "audit_required": decision == "AUDIT",
            "reject": decision == "REJECT"
        }
    
    def _determine_decision(self, score: float) -> str:
        """
        Determine decision based on score thresholds.
        
        Args:
            score: Hybrid similarity score
            
        Returns:
            Decision string (AUTO_MERGE, AUDIT, REJECT)
        """
        auto_merge_threshold = self.thresholds.get("auto_merge", 0.85)
        audit_low_threshold = self.thresholds.get("audit_low", 0.65)
        audit_high_threshold = self.thresholds.get("audit_high", 0.85)
        
        if score >= auto_merge_threshold:
            return "AUTO_MERGE"
        elif audit_low_threshold <= score < audit_high_threshold:
            return "AUDIT"
        else:
            return "REJECT"
    
    def score_candidate_pairs(self, scored_pairs_df: pd.DataFrame, 
                            candidates_df: pd.DataFrame,
                            providers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all candidate pairs using hybrid approach.
        
        Args:
            scored_pairs_df: DataFrame with deterministic scores
            candidates_df: DataFrame with candidate pairs
            providers_df: DataFrame with provider records
            
        Returns:
            DataFrame with hybrid scores and decisions
        """
        logger.info(f"Scoring {len(scored_pairs_df)} pairs with hybrid approach")
        
        # Get ML predictions if model is available
        ml_predictions_df = None
        if self.ml_predictor:
            ml_predictions_df = self.ml_predictor.predict_pairs(candidates_df, providers_df)
        
        # Calculate hybrid scores
        hybrid_results = []
        
        for idx, row in scored_pairs_df.iterrows():
            # Extract deterministic scores
            deterministic_scores = {
                "name_exact_match": row.get("name_exact_match", 0.0),
                "name_levenshtein_match": row.get("name_levenshtein_match", 0.0),
                "name_fuzz_ratio_match": row.get("name_fuzz_ratio_match", 0.0),
                "affiliation_jaccard_similarity": row.get("affiliation_jaccard_similarity", 0.0),
                "location_location_match": row.get("location_location_match", 0.0),
                "contact_contact_match": row.get("contact_contact_match", 0.0)
            }
            
            # Get ML probability if available
            ml_probability = None
            if ml_predictions_df is not None and idx < len(ml_predictions_df):
                ml_probability = ml_predictions_df.iloc[idx].get("match_probability", 0.0)
            
            # Calculate hybrid score
            hybrid_score = self.calculate_hybrid_score(deterministic_scores, ml_probability)
            
            # Create result row
            result = {
                "record_id_1": row.get("record_id_1"),
                "record_id_2": row.get("record_id_2"),
                "block_key": row.get("block_key", ""),
                "strategy": row.get("strategy", ""),
                **deterministic_scores,
                **hybrid_score
            }
            
            hybrid_results.append(result)
        
        hybrid_df = pd.DataFrame(hybrid_results)
        
        logger.info(f"Hybrid scoring completed: {len(hybrid_df)} pairs scored")
        return hybrid_df
    
    def get_scoring_statistics(self, hybrid_df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate comprehensive scoring statistics.
        
        Args:
            hybrid_df: DataFrame with hybrid scores
            
        Returns:
            Dictionary with scoring statistics
        """
        if "hybrid_score" not in hybrid_df.columns:
            return {}
        
        scores = hybrid_df["hybrid_score"]
        
        # Decision distribution
        decision_counts = hybrid_df["decision"].value_counts().to_dict()
        
        # Score distribution
        score_stats = {
            "mean_score": scores.mean(),
            "median_score": scores.median(),
            "std_score": scores.std(),
            "min_score": scores.min(),
            "max_score": scores.max()
        }
        
        # Threshold statistics
        auto_merge_threshold = self.thresholds.get("auto_merge", 0.85)
        audit_low_threshold = self.thresholds.get("audit_low", 0.65)
        
        threshold_stats = {
            "auto_merge_count": (scores >= auto_merge_threshold).sum(),
            "audit_count": ((scores >= audit_low_threshold) & (scores < auto_merge_threshold)).sum(),
            "reject_count": (scores < audit_low_threshold).sum(),
            "auto_merge_percentage": (scores >= auto_merge_threshold).sum() / len(scores) * 100,
            "audit_percentage": ((scores >= audit_low_threshold) & (scores < auto_merge_threshold)).sum() / len(scores) * 100,
            "reject_percentage": (scores < audit_low_threshold).sum() / len(scores) * 100
        }
        
        statistics = {
            "total_pairs": len(hybrid_df),
            "score_statistics": score_stats,
            "decision_distribution": decision_counts,
            "threshold_statistics": threshold_stats,
            "thresholds": self.thresholds
        }
        
        return statistics
    
    def optimize_thresholds(self, validation_df: pd.DataFrame, 
                          target_precision: float = 0.92,
                          target_recall: float = 0.88) -> Dict[str, float]:
        """
        Optimize decision thresholds based on validation data.
        
        Args:
            validation_df: Validation DataFrame with true labels
            target_precision: Target precision score
            target_recall: Target recall score
            
        Returns:
            Dictionary with optimized thresholds
        """
        if "hybrid_score" not in validation_df.columns or "label" not in validation_df.columns:
            logger.warning("Validation data missing required columns")
            return self.thresholds
        
        scores = validation_df["hybrid_score"].values
        labels = validation_df["label"].values
        
        # Find optimal thresholds
        best_thresholds = self.thresholds.copy()
        best_f1 = 0.0
        
        # Search for optimal auto_merge threshold
        for threshold in np.arange(0.5, 0.95, 0.01):
            predictions = (scores >= threshold).astype(int)
            
            # Calculate metrics
            tp = ((predictions == 1) & (labels == 1)).sum()
            fp = ((predictions == 1) & (labels == 0)).sum()
            fn = ((predictions == 0) & (labels == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Check if meets targets and improves F1
            if precision >= target_precision and recall >= target_recall and f1 > best_f1:
                best_f1 = f1
                best_thresholds["auto_merge"] = threshold
                best_thresholds["audit_high"] = threshold
        
        # Set audit low threshold slightly below auto_merge
        best_thresholds["audit_low"] = max(0.5, best_thresholds["auto_merge"] - 0.2)
        
        logger.info(f"Optimized thresholds: auto_merge={best_thresholds['auto_merge']:.3f}, "
                   f"audit_low={best_thresholds['audit_low']:.3f}, "
                   f"audit_high={best_thresholds['audit_high']:.3f}")
        
        return best_thresholds
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update scoring weights.
        
        Args:
            new_weights: New weight dictionary
        """
        self.weights.update(new_weights)
        logger.info(f"Updated scoring weights: {self.weights}")
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """
        Update decision thresholds.
        
        Args:
            new_thresholds: New threshold dictionary
        """
        self.thresholds.update(new_thresholds)
        logger.info(f"Updated decision thresholds: {self.thresholds}")


def create_hybrid_scorer(config_path: str = "config/provider_verify.yaml",
                        ml_model_path: Optional[str] = None) -> HybridScorer:
    """
    Convenience function to create hybrid scorer.
    
    Args:
        config_path: Path to configuration file
        ml_model_path: Path to ML model (optional)
        
    Returns:
        Initialized hybrid scorer
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        scoring_config = config.get("scoring", {})
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        scoring_config = {}
    
    return HybridScorer(scoring_config, ml_model_path)


def apply_hybrid_scoring(scored_pairs_df: pd.DataFrame,
                        candidates_df: pd.DataFrame,
                        providers_df: pd.DataFrame,
                        config_path: str = "config/provider_verify.yaml",
                        ml_model_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to apply hybrid scoring to candidate pairs.
    
    Args:
        scored_pairs_df: DataFrame with deterministic scores
        candidates_df: DataFrame with candidate pairs
        providers_df: DataFrame with provider records
        config_path: Path to configuration file
        ml_model_path: Path to ML model (optional)
        
    Returns:
        DataFrame with hybrid scores and decisions
    """
    scorer = create_hybrid_scorer(config_path, ml_model_path)
    hybrid_df = scorer.score_candidate_pairs(scored_pairs_df, candidates_df, providers_df)
    
    # Get statistics
    stats = scorer.get_scoring_statistics(hybrid_df)
    logger.info(f"Hybrid scoring completed: {stats['threshold_statistics']['auto_merge_count']} auto-merge, "
               f"{stats['threshold_statistics']['audit_count']} audit, "
               f"{stats['threshold_statistics']['reject_count']} reject")
    
    return hybrid_df
