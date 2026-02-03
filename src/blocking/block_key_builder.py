"""
Block key builder for ProviderVerify.

Generates deterministic block keys for multi-layer blocking strategies
to dramatically reduce candidate pairs for similarity scoring.
"""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
from collections import defaultdict
from itertools import combinations

logger = logging.getLogger(__name__)


class BlockKeyBuilder:
    """
    Builds block keys for multi-layer deterministic blocking strategies.
    
    Reduces pairwise comparisons from billions to millions by grouping
    similar records into blocks for candidate generation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize block key builder with configuration.
        
        Args:
            config: Configuration dictionary with blocking strategies
        """
        self.config = config
        self.strategies = config.get("strategies", [])
        self.max_candidates_per_block = config.get("max_candidates_per_block", 1000)
        
        logger.info(f"Initialized BlockKeyBuilder with {len(self.strategies)} blocking strategies")
    
    def generate_block_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate block keys for all records using configured strategies.
        
        Args:
            df: Normalized provider DataFrame
            
        Returns:
            DataFrame with block keys for each strategy
        """
        result_df = df.copy()
        
        for strategy in self.strategies:
            strategy_name = strategy.get("name", "unnamed")
            keys = strategy.get("keys", [])
            similarity_threshold = strategy.get("similarity_threshold")
            
            logger.info(f"Generating block keys for strategy: {strategy_name}")
            
            if similarity_threshold:
                # Token-based blocking with similarity threshold
                block_column = f"block_{strategy_name}"
                result_df[block_column] = self._generate_token_based_blocks(
                    df, keys, similarity_threshold
                )
            else:
                # Deterministic blocking
                block_column = f"block_{strategy_name}"
                result_df[block_column] = self._generate_deterministic_blocks(df, keys)
        
        logger.info(f"Generated block keys for {len(result_df)} records")
        return result_df
    
    def _generate_deterministic_blocks(self, df: pd.DataFrame, keys: List[str]) -> pd.Series:
        """
        Generate deterministic block keys based on specified fields.
        
        Args:
            df: Input DataFrame
            keys: List of field names for block key generation
            
        Returns:
            Series with block keys
        """
        block_keys = []
        
        for _, row in df.iterrows():
            key_components = []
            
            for key in keys:
                # Handle field expressions like "zip_norm[:3]"
                field_value = self._extract_field_value(row, key)
                if field_value:
                    key_components.append(str(field_value))
            
            # Create block key from components
            if key_components:
                block_key = "|".join(key_components)
                # Hash for consistent length keys
                block_key_hash = hashlib.md5(block_key.encode()).hexdigest()[:16]
                block_keys.append(block_key_hash)
            else:
                block_keys.append("")  # Empty block key for records with missing data
        
        return pd.Series(block_keys, index=df.index)
    
    def _generate_token_based_blocks(self, df: pd.DataFrame, keys: List[str], 
                                   similarity_threshold: float) -> pd.Series:
        """
        Generate token-based block keys with similarity threshold.
        
        Args:
            df: Input DataFrame
            keys: List of field names for token extraction
            similarity_threshold: Minimum similarity for blocking
            
        Returns:
            Series with block keys
        """
        block_keys = []
        
        for _, row in df.iterrows():
            # Extract tokens from specified fields
            all_tokens = set()
            
            for key in keys:
                field_value = self._extract_field_value(row, key)
                if field_value and isinstance(field_value, str):
                    tokens = self._extract_tokens(field_value)
                    all_tokens.update(tokens)
            
            # Create block keys from token combinations
            if all_tokens:
                # Use sorted tokens for consistency
                sorted_tokens = sorted(all_tokens)
                
                # Create block key from first few tokens
                num_tokens = min(3, len(sorted_tokens))  # Use up to 3 tokens
                key_tokens = sorted_tokens[:num_tokens]
                
                block_key = "_".join(key_tokens)
                block_key_hash = hashlib.md5(block_key.encode()).hexdigest()[:16]
                block_keys.append(block_key_hash)
            else:
                block_keys.append("")
        
        return pd.Series(block_keys, index=df.index)
    
    def _extract_field_value(self, row: pd.Series, field_expression: str) -> str:
        """
        Extract field value from row, handling expressions like "zip_norm[:3]".
        
        Args:
            row: DataFrame row
            field_expression: Field expression (e.g., "zip_norm[:3]")
            
        Returns:
            Extracted field value
        """
        # Parse field expression
        if "[" in field_expression and "]" in field_expression:
            # Handle slicing expressions like "zip_norm[:3]"
            field_name = field_expression.split("[")[0]
            slice_expr = field_expression.split("[")[1].rstrip("]")
            
            field_value = row.get(field_name, "")
            if field_value and isinstance(field_value, str):
                try:
                    # Evaluate slice expression safely
                    if slice_expr.endswith(":3"):
                        return field_value[:3]
                    elif slice_expr.endswith(":5]"):
                        return field_value[:5]
                    else:
                        return field_value
                except:
                    return field_value
            else:
                return ""
        else:
            # Simple field reference
            return row.get(field_expression, "")
    
    def _extract_tokens(self, text: str) -> List[str]:
        """
        Extract tokens from text for token-based blocking.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        # Split on whitespace and punctuation
        tokens = []
        for token in text.lower().split():
            # Remove punctuation and normalize
            clean_token = "".join(c for c in token if c.isalnum())
            if len(clean_token) >= 2:  # Only keep tokens with 2+ characters
                tokens.append(clean_token)
        
        return tokens
    
    def generate_candidate_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate candidate pairs using all blocking strategies.
        
        Args:
            df: DataFrame with block keys
            
        Returns:
            DataFrame with candidate pairs (record_id_1, record_id_2, block_key, strategy)
        """
        all_candidates = []
        
        # Get block key columns
        block_columns = [col for col in df.columns if col.startswith("block_")]
        
        for block_column in block_columns:
            strategy_name = block_column.replace("block_", "")
            logger.info(f"Generating candidates for strategy: {strategy_name}")
            
            # Group records by block key
            block_groups = df.groupby(block_column)
            
            for block_key, group in block_groups:
                if not block_key:  # Skip empty block keys
                    continue
                
                # Limit block size to prevent memory issues
                if len(group) > self.max_candidates_per_block:
                    logger.warning(f"Block {block_key} too large ({len(group)} records), skipping")
                    continue
                
                # Generate all pairs within the block
                record_ids = group.index.tolist()
                
                for id1, id2 in combinations(record_ids, 2):
                    candidate = {
                        "record_id_1": id1,
                        "record_id_2": id2,
                        "block_key": block_key,
                        "strategy": strategy_name
                    }
                    all_candidates.append(candidate)
        
        if not all_candidates:
            logger.warning("No candidate pairs generated")
            return pd.DataFrame(columns=["record_id_1", "record_id_2", "block_key", "strategy"])
        
        candidates_df = pd.DataFrame(all_candidates)
        
        # Remove duplicate pairs (same pair from different strategies)
        candidates_df["pair_key"] = candidates_df.apply(
            lambda row: tuple(sorted([row["record_id_1"], row["record_id_2"]])), axis=1
        )
        candidates_df = candidates_df.drop_duplicates(subset=["pair_key"], keep="first")
        candidates_df = candidates_df.drop("pair_key", axis=1)
        
        logger.info(f"Generated {len(candidates_df)} unique candidate pairs")
        return candidates_df
    
    def get_blocking_statistics(self, df: pd.DataFrame, 
                              candidates_df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate blocking efficiency statistics.
        
        Args:
            df: Original DataFrame with block keys
            candidates_df: Candidate pairs DataFrame
            
        Returns:
            Dictionary with blocking statistics
        """
        total_records = len(df)
        total_possible_pairs = total_records * (total_records - 1) // 2
        generated_candidates = len(candidates_df)
        
        # Calculate reduction ratio
        reduction_ratio = 1 - (generated_candidates / total_possible_pairs) if total_possible_pairs > 0 else 0
        
        # Block statistics
        block_columns = [col for col in df.columns if col.startswith("block_")]
        block_stats = {}
        
        for block_column in block_columns:
            strategy_name = block_column.replace("block_", "")
            non_empty_blocks = df[df[block_column] != ""]
            
            if len(non_empty_blocks) > 0:
                avg_block_size = len(non_empty_blocks) / non_empty_blocks[block_column].nunique()
                max_block_size = non_empty_blocks.groupby(block_column).size().max()
                
                block_stats[strategy_name] = {
                    "num_blocks": non_empty_blocks[block_column].nunique(),
                    "avg_block_size": avg_block_size,
                    "max_block_size": max_block_size,
                    "coverage": len(non_empty_blocks) / total_records
                }
        
        statistics = {
            "total_records": total_records,
            "total_possible_pairs": total_possible_pairs,
            "generated_candidates": generated_candidates,
            "reduction_ratio": reduction_ratio,
            "reduction_percentage": reduction_ratio * 100,
            "blocking_strategies": block_stats
        }
        
        logger.info(f"Blocking statistics: {generated_candidates:,} candidates from "
                   f"{total_possible_pairs:,} possible pairs "
                   f"({reduction_percentage:.2f}% reduction)")
        
        return statistics


def create_blocking_keys(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to create block keys and generate candidate pairs.
    
    Args:
        df: Normalized provider DataFrame
        config: Blocking configuration
        
    Returns:
        Tuple of (df_with_block_keys, candidate_pairs_df)
    """
    builder = BlockKeyBuilder(config)
    
    # Generate block keys
    df_with_keys = builder.generate_block_keys(df)
    
    # Generate candidate pairs
    candidates_df = builder.generate_candidate_pairs(df_with_keys)
    
    # Get statistics
    stats = builder.get_blocking_statistics(df_with_keys, candidates_df)
    logger.info(f"Blocking completed: {stats['generated_candidates']:,} candidate pairs "
               f"({stats['reduction_percentage']:.1f}% reduction)")
    
    return df_with_keys, candidates_df
