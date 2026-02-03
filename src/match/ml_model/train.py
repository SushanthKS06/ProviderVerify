"""
Training script for ProviderVerify ML matching model.

Trains an XGBoost classifier on labeled provider pairs to predict
match probability and enhance deterministic scoring.
"""

import logging
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import yaml

logger = logging.getLogger(__name__)


class ProviderMatchModelTrainer:
    """
    Trains machine learning model for provider matching.
    
    Uses XGBoost classifier on engineered features from provider pairs
    to predict match probability with high accuracy.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize model trainer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.algorithm = config.get("algorithm", "xgboost")
        self.model_path = config.get("model_path", "models/provider_match.xgb")
        self.feature_columns = config.get("feature_columns", [])
        self.training_config = config.get("training", {})
        
        # Initialize model
        if self.algorithm == "xgboost":
            self.model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=self.training_config.get("random_state", 42)
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Initialize feature scaler
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized {self.algorithm} model trainer")
    
    def load_training_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare training data from CSV file.
        
        Args:
            data_path: Path to training data CSV
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded training data from {data_path}: {len(df)} records")
            
            # Validate required columns
            required_columns = ["pair_id", "label"] + self.feature_columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Extract features and labels
            features_df = df[self.feature_columns].copy()
            labels_series = df["label"].copy()
            
            # Handle missing values
            features_df = features_df.fillna(0)
            
            # Validate labels
            unique_labels = labels_series.unique()
            if not all(label in [0, 1] for label in unique_labels):
                raise ValueError(f"Labels must be 0 or 1, found: {unique_labels}")
            
            logger.info(f"Training data: {len(features_df)} samples, "
                       f"{len(self.feature_columns)} features, "
                       f"{labels_series.sum()} positive examples")
            
            return features_df, labels_series
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise
    
    def engineer_features(self, rec1: Dict, rec2: Dict) -> np.ndarray:
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
    
    def train_model(self, features_df: pd.DataFrame, labels_series: pd.Series) -> Dict[str, float]:
        """
        Train the ML model on provided data.
        
        Args:
            features_df: Feature DataFrame
            labels_series: Label Series
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        test_size = self.training_config.get("test_size", 0.2)
        random_state = self.training_config.get("random_state", 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels_series, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels_series
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning
        if self.training_config.get("hyperparameter_tuning", False):
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            cv_folds = self.training_config.get("cv_folds", 5)
            grid_search = GridSearchCV(
                self.model, param_grid, 
                cv=cv_folds, 
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            # Train with default parameters
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        
        metrics = {
            "auc_score": auc_score,
            "optimal_threshold": optimal_threshold,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": X_train.shape[1]
        }
        
        logger.info(f"Model training completed: AUC = {auc_score:.4f}, "
                   f"Optimal threshold = {optimal_threshold:.3f}")
        
        return metrics
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save trained model and scaler to disk.
        
        Args:
            save_path: Custom save path (optional)
            
        Returns:
            Path where model was saved
        """
        if save_path is None:
            save_path = self.model_path
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "algorithm": self.algorithm,
            "config": self.config
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {save_path}")
        return save_path
    
    def cross_validate(self, features_df: pd.DataFrame, labels_series: pd.Series) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            features_df: Feature DataFrame
            labels_series: Label Series
            
        Returns:
            Dictionary with cross-validation metrics
        """
        cv_folds = self.training_config.get("cv_folds", 5)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features_df)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, labels_series, 
            cv=cv_folds, 
            scoring='roc_auc'
        )
        
        cv_metrics = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_min": cv_scores.min(),
            "cv_max": cv_scores.max()
        }
        
        logger.info(f"Cross-validation completed: {cv_metrics['cv_mean']:.4f} ± {cv_metrics['cv_std']:.4f}")
        
        return cv_metrics


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train ProviderVerify ML model")
    parser.add_argument("--data", required=True, help="Path to training data CSV")
    parser.add_argument("--config", default="config/provider_verify.yaml", help="Path to config file")
    parser.add_argument("--output", help="Output model path")
    parser.add_argument("--cv", action="store_true", help="Perform cross-validation")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        ml_config = config.get("ml_model", {})
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
    
    # Initialize trainer
    trainer = ProviderMatchModelTrainer(ml_config)
    
    # Load training data
    features_df, labels_series = trainer.load_training_data(args.data)
    
    # Train model
    metrics = trainer.train_model(features_df, labels_series)
    
    # Cross-validation if requested
    if args.cv:
        cv_metrics = trainer.cross_validate(features_df, labels_series)
        metrics.update(cv_metrics)
    
    # Save model
    model_path = trainer.save_model(args.output)
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Model saved to: {model_path}")
    print(f"AUC Score: {metrics['auc_score']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.3f}")
    if 'cv_mean' in metrics:
        print(f"CV AUC: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")


if __name__ == "__main__":
    main()
