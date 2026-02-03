"""
Configuration utilities for normalization modules.

Provides configuration loading and validation for all normalization components.
"""

import logging
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_normalization_config(config_path: str = "config/provider_verify.yaml") -> Dict[str, Any]:
    """
    Load normalization configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Configuration file {config_path} not found, using defaults")
            return get_default_normalization_config()
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract normalization section
        norm_config = config.get('normalization', {})
        
        # Add scoring and blocking configs for similarity calculations
        norm_config['scoring'] = config.get('scoring', {})
        norm_config['blocking'] = config.get('blocking', {})
        
        logger.info(f"Loaded normalization configuration from {config_path}")
        return norm_config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        return get_default_normalization_config()


def get_default_normalization_config() -> Dict[str, Any]:
    """
    Get default normalization configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "name": {
            "remove_titles": ["Dr.", "Dr", "MD", "PhD", "DO", "PA", "NP", "RN"],
            "remove_suffixes": ["Jr.", "Sr.", "II", "III", "IV"],
            "case_format": "title"
        },
        "affiliation": {
            "min_similarity_threshold": 0.5,
            "master_affiliation_file": "config/master_affiliations.csv"
        },
        "address": {
            "standardize_city_state": True,
            "normalize_street": True,
            "zip_regex": "\\d{5}(-\\d{4})?"
        },
        "phone": {
            "default_country_code": "US",
            "format": "E164"
        },
        "email": {
            "case_sensitive": False,
            "validate_format": True
        },
        "scoring": {
            "fuzzy_thresholds": {
                "levenshtein_max_distance": 2,
                "fuzz_ratio_min": 90,
                "jaccard_min_similarity": 0.3
            }
        },
        "blocking": {
            "max_candidates_per_block": 1000
        }
    }


def validate_normalization_config(config: Dict[str, Any]) -> bool:
    """
    Validate normalization configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_sections = ["name", "affiliation", "address"]
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate name configuration
    name_config = config.get("name", {})
    if not isinstance(name_config.get("remove_titles", []), list):
        logger.error("name.remove_titles must be a list")
        return False
    
    if not isinstance(name_config.get("remove_suffixes", []), list):
        logger.error("name.remove_suffixes must be a list")
        return False
    
    # Validate affiliation configuration
    affiliation_config = config.get("affiliation", {})
    min_threshold = affiliation_config.get("min_similarity_threshold", 0.5)
    if not isinstance(min_threshold, (int, float)) or not 0 <= min_threshold <= 1:
        logger.error("affiliation.min_similarity_threshold must be a number between 0 and 1")
        return False
    
    # Validate address configuration
    address_config = config.get("address", {})
    if not isinstance(address_config.get("standardize_city_state", True), bool):
        logger.error("address.standardize_city_state must be a boolean")
        return False
    
    logger.info("Configuration validation passed")
    return True


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def save_normalization_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save normalization configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        return False
