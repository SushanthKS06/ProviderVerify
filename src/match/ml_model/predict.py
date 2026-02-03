"""
Prediction module for ProviderVerify ML matching model.

Loads trained model and predicts match probability for provider pairs
to enhance deterministic scoring with machine learning insights.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ProviderMatchPredictor:
    """
    Predicts match probability for provider pairs using trained ML model.
    
    Enhances deterministic scoring with data-driven probability estimates
    for more accurate entity resolution.
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained model file
            config_path: Path to configuration file (optional)
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.algorithm = ""
        self.config = {}
        
        # Load model
        self._load_model()
        
        # Load configuration if provided
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        logger.info(f"Initialized {self.algorithm} predictor from {model_path}")
    
    def _load_model(self):
        """Load trained model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_columns = model_data["feature_columns"]
            self.algorithm = model_data["algorithm"]
            self.config = model_data.get("config", {})
            
            logger.info(f"Loaded {self.algorithm} model with {len(self.feature_columns)} features")
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def predict_pair(self, rec1: Dict, rec2: Dict) -> Dict[str, float]:
        """
        Predict match probability for a single provider pair.
        
        Args:
            rec1: First provider record
            rec2: Second provider record
            
        Returns:
            Dictionary with prediction results
        """
        # Engineer features
        features = self._engineer_features(rec1, rec2)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict probability
        match_probability = self.model.predict_proba(features_scaled)[0, 1]
        
        # Predict class (using default threshold of 0.5)
        match_prediction = int(match_probability >= 0.5)
        
        return {
            "match_probability": match_probability,
            "match_prediction": match_prediction,
            "features": dict(zip(self.feature_columns, features))
        }
    
    def predict_pairs(self, candidates_df: pd.DataFrame, 
                     providers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict match probability for multiple provider pairs.
        
        Args:
            candidates_df: DataFrame with candidate pairs
            providers_df: DataFrame with provider records
            
        Returns:
            DataFrame with prediction results
        """
        predictions = []
        
        for _, pair in candidates_df.iterrows():
            id1 = pair["record_id_1"]
            id2 = pair["record_id_2"]
            
            # Get provider records
            rec1 = providers_df.loc[id1].to_dict()
            rec2 = providers_df.loc[id2].to_dict()
            
            # Predict
            result = self.predict_pair(rec1, rec2)
            
            # Add pair information
            prediction_data = {
                "record_id_1": id1,
                "record_id_2": id2,
                **result
            }
            
            predictions.append(prediction_data)
        
        predictions_df = pd.DataFrame(predictions)
        
        logger.info(f"Generated predictions for {len(predictions_df)} pairs")
        return predictions_df
    
    def _engineer_features(self, rec1: Dict, rec2: Dict) -> np.ndarray:
        """
        Engineer features for a single provider pair.
        
        Args:
            rec1: First provider record
            rec2: Second provider record
            
        Returns:
            Feature array
        """
        features = []
        
        # Name similarity features
        name1 = rec1.get("norm_name", "")
        name2 = rec2.get("norm_name", "")
        features.extend(self._calculate_name_features(name1, name2))
        
        # Affiliation similarity features
        aff1 = rec1.get("norm_affiliation", "")
        aff2 = rec2.get("norm_affiliation", "")
        features.extend(self._calculate_affiliation_features(aff1, aff2))
        
        # Location features
        features.extend(self._calculate_location_features(rec1, rec2))
        
        # Contact features
        features.extend(self._calculate_contact_features(rec1, rec2))
        
        # Additional features
        features.extend(self._calculate_additional_features(rec1, rec2))
        
        return np.array(features)
    
    def _calculate_name_features(self, name1: str, name2: str) -> List[float]:
        """Calculate name similarity features."""
        from thefuzz import fuzz
        from Levenshtein import distance as levenshtein_distance
        from jellyfish import jaro_winkler_similarity
        
        if not name1 or not name2:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Exact match
        exact_match = 1.0 if name1.lower() == name2.lower() else 0.0
        
        # Levenshtein distance
        lev_dist = levenshtein_distance(name1.lower(), name2.lower())
        lev_normalized = lev_dist / max(len(name1), len(name2))
        
        # Fuzzy ratio
        fuzz_ratio = fuzz.ratio(name1, name2) / 100.0
        
        # Jaro-Winkler similarity
        jaro_winkler = jaro_winkler_similarity(name1.lower(), name2.lower())
        
        # Length difference
        length_diff = abs(len(name1) - len(name2)) / max(len(name1), len(name2))
        
        return [exact_match, lev_normalized, fuzz_ratio, jaro_winkler, length_diff]
    
    def _calculate_affiliation_features(self, aff1: str, aff2: str) -> List[float]:
        """Calculate affiliation similarity features."""
        from thefuzz import fuzz
        
        if not aff1 or not aff2:
            return [0.0, 0.0, 0.0]
        
        # Exact match
        exact_match = 1.0 if aff1.lower() == aff2.lower() else 0.0
        
        # Token Jaccard similarity
        tokens1 = set(aff1.lower().split())
        tokens2 = set(aff2.lower().split())
        jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
        
        # Fuzzy ratio
        fuzz_ratio = fuzz.ratio(aff1, aff2) / 100.0
        
        return [exact_match, jaccard, fuzz_ratio]
    
    def _calculate_location_features(self, rec1: Dict, rec2: Dict) -> List[float]:
        """Calculate location similarity features."""
        features = []
        
        # City match
        city1 = rec1.get("city_norm", "").lower()
        city2 = rec2.get("city_norm", "").lower()
        features.append(1.0 if city1 == city2 and city1 != "" else 0.0)
        
        # State match
        state1 = rec1.get("state_norm", "").upper()
        state2 = rec2.get("state_norm", "").upper()
        features.append(1.0 if state1 == state2 and state1 != "" else 0.0)
        
        # ZIP match
        zip1 = rec1.get("zip_norm", "")
        zip2 = rec2.get("zip_norm", "")
        features.append(1.0 if zip1 == zip2 and zip1 != "" else 0.0)
        
        # Street match
        street1 = rec1.get("address_line_1_norm", "").lower()
        street2 = rec2.get("address_line_1_norm", "").lower()
        features.append(1.0 if street1 == street2 and street1 != "" else 0.0)
        
        return features
    
    def _calculate_contact_features(self, rec1: Dict, rec2: Dict) -> List[float]:
        """Calculate contact similarity features."""
        # Phone match
        phone1 = rec1.get("phone_norm", "")
        phone2 = rec2.get("phone_norm", "")
        phone_match = 1.0 if phone1 == phone2 and phone1 != "" else 0.0
        
        # Email match
        email1 = rec1.get("email_norm", "").lower()
        email2 = rec2.get("email_norm", "").lower()
        email_match = 1.0 if email1 == email2 and email1 != "" else 0.0
        
        return [phone_match, email_match]
    
    def _calculate_additional_features(self, rec1: Dict, rec2: Dict) -> List[float]:
        """Calculate additional features."""
        features = []
        
        # Source match (same data source)
        source1 = rec1.get("source", "")
        source2 = rec2.get("source", "")
        features.append(1.0 if source1 == source2 and source1 != "" else 0.0)
        
        # Provider ID similarity (if similar patterns)
        id1 = rec1.get("provider_id", "")
        id2 = rec2.get("provider_id", "")
        id_similarity = 1.0 if id1 == id2 and id1 != "" else 0.0
        features.append(id_similarity)
        
        return features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model.
        
        Returns:
            Dictionary with feature importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not support feature importance")
            return {}
        
        importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def get_prediction_statistics(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistics for predictions.
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            Dictionary with prediction statistics
        """
        if "match_probability" not in predictions_df.columns:
            return {}
        
        probabilities = predictions_df["match_probability"]
        
        stats = {
            "mean_probability": probabilities.mean(),
            "median_probability": probabilities.median(),
            "std_probability": probabilities.std(),
            "min_probability": probabilities.min(),
            "max_probability": probabilities.max(),
            "high_confidence_matches": (probabilities >= 0.9).sum(),
            "low_confidence_matches": (probabilities <= 0.1).sum(),
            "uncertain_predictions": ((probabilities > 0.4) & (probabilities < 0.6)).sum()
        }
        
        return stats
    
    def calibrate_threshold(self, validation_data: Tuple[pd.DataFrame, pd.Series], 
                           target_metric: str = "f1") -> float:
        """
        Calibrate prediction threshold based on validation data.
        
        Args:
            validation_data: Tuple of (features_df, labels_series)
            target_metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold value
        """
        from sklearn.metrics import precision_recall_curve, f1_score
        
        features_df, labels_series = validation_data
        
        # Scale features
        X_scaled = self.scaler.transform(features_df)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(labels_series, probabilities)
        
        if target_metric == "f1":
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
        elif target_metric == "precision":
            optimal_idx = np.argmax(precision)
        elif target_metric == "recall":
            optimal_idx = np.argmax(recall)
        else:
            raise ValueError(f"Unsupported target metric: {target_metric}")
        
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"Optimal threshold for {target_metric}: {optimal_threshold:.3f}")
        
        return optimal_threshold


def load_predictor(model_path: str, config_path: Optional[str] = None) -> ProviderMatchPredictor:
    """
    Convenience function to load predictor.
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file (optional)
        
    Returns:
        Initialized predictor
    """
    return ProviderMatchPredictor(model_path, config_path)


def predict_matches(candidates_df: pd.DataFrame, 
                   providers_df: pd.DataFrame,
                   model_path: str,
                   config_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to predict matches for candidate pairs.
    
    Args:
        candidates_df: Candidate pairs DataFrame
        providers_df: Provider records DataFrame
        model_path: Path to trained model
        config_path: Path to configuration file (optional)
        
    Returns:
        DataFrame with predictions
    """
    predictor = load_predictor(model_path, config_path)
    return predictor.predict_pairs(candidates_df, providers_df)
