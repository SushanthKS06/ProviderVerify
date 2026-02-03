"""
Provider record merger for ProviderVerify.

Merges duplicate provider records into canonical entities with conflict
resolution, provenance tracking, and canonical ID generation.
"""

import hashlib
import logging
import uuid
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProviderMerger:
    """
    Merges duplicate provider records into canonical entities.
    
    Handles conflict resolution, provenance tracking, and maintains
    data quality during the merge process.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize provider merger with configuration.
        
        Args:
            config: Configuration dictionary with merge rules
        """
        self.config = config
        self.merge_config = config.get("merge", {})
        self.conflict_resolution = self.merge_config.get("conflict_resolution", {})
        self.canonical_config = self.merge_config.get("canonical_id", {})
        
        # Set default conflict resolution strategy
        self.default_conflict_strategy = self.conflict_resolution.get("strategy", "most_complete")
        self.keep_all_values = self.conflict_resolution.get("keep_all_values", True)
        self.provenance_tags = self.conflict_resolution.get("provenance_tags", True)
        
        logger.info("Initialized ProviderMerger")
    
    def generate_canonical_id(self, record: Dict) -> str:
        """
        Generate canonical ID for a provider record.
        
        Args:
            record: Provider record dictionary
            
        Returns:
            Canonical ID string
        """
        algorithm = self.canonical_config.get("algorithm", "uuid4_sha256")
        hash_fields = self.canonical_config.get("hash_fields", ["norm_name", "city_norm", "state_norm"])
        
        if algorithm == "uuid4_sha256":
            # Create hash from specified fields
            hash_components = []
            for field in hash_fields:
                value = record.get(field, "")
                if value:
                    hash_components.append(str(value).lower().strip())
            
            if hash_components:
                hash_string = "|".join(hash_components)
                sha256_hash = hashlib.sha256(hash_string.encode()).hexdigest()[:16]
                canonical_id = f"PROV-{sha256_hash.upper()}"
            else:
                # Fallback to UUID4
                canonical_id = f"PROV-{uuid.uuid4().hex[:16].upper()}"
        
        elif algorithm == "uuid4":
            canonical_id = f"PROV-{uuid.uuid4().hex[:16].upper()}"
        
        else:
            # Default to UUID4
            canonical_id = f"PROV-{uuid.uuid4().hex[:16].upper()}"
        
        return canonical_id
    
    def resolve_field_conflict(self, field_name: str, values: List[Dict]) -> any:
        """
        Resolve conflicts between multiple field values.
        
        Args:
            field_name: Name of the field
            values: List of tuples (value, source_record_id)
            
        Returns:
            Resolved value
        """
        if not values:
            return ""
        
        if len(values) == 1:
            return values[0]["value"]
        
        # Extract just the values for comparison
        value_list = [v["value"] for v in values]
        
        # Check if all values are the same
        if len(set(str(v) for v in value_list if v)) == 1:
            return values[0]["value"]
        
        # Apply conflict resolution strategy
        if self.default_conflict_strategy == "most_complete":
            return self._resolve_most_complete(field_name, values)
        elif self.default_conflict_strategy == "longest":
            return self._resolve_longest(field_name, values)
        elif self.default_conflict_strategy == "most_recent":
            return self._resolve_most_recent(field_name, values)
        else:
            # Default to first value
            return values[0]["value"]
    
    def _resolve_most_complete(self, field_name: str, values: List[Dict]) -> any:
        """Resolve conflict by choosing most complete (non-empty) value."""
        # Filter out empty values
        non_empty_values = [v for v in values if v["value"] and str(v["value"]).strip()]
        
        if not non_empty_values:
            return ""
        
        # For text fields, choose the longest non-empty value
        if field_name in ["raw_name", "norm_name", "affiliation", "norm_affiliation", 
                         "address", "address_line_1_norm"]:
            return max(non_empty_values, key=lambda x: len(str(x["value"])))["value"]
        
        # For other fields, choose the first non-empty value
        return non_empty_values[0]["value"]
    
    def _resolve_longest(self, field_name: str, values: List[Dict]) -> any:
        """Resolve conflict by choosing longest value."""
        return max(values, key=lambda x: len(str(x["value"]) if x["value"] else ""))["value"]
    
    def _resolve_most_recent(self, field_name: str, values: List[Dict]) -> any:
        """Resolve conflict by choosing most recent (placeholder)."""
        # In a real implementation, this would use timestamps
        # For now, return the first value
        return values[0]["value"]
    
    def merge_records(self, records: List[Dict], record_ids: List[int]) -> Dict:
        """
        Merge multiple provider records into a canonical record.
        
        Args:
            records: List of provider records to merge
            record_ids: List of original record IDs
            
        Returns:
            Merged canonical record
        """
        if not records:
            return {}
        
        if len(records) == 1:
            # Single record, just add merge metadata
            merged = records[0].copy()
            merged["canonical_id"] = self.generate_canonical_id(records[0])
            merged["merged_from"] = [record_ids[0]]
            merged["merge_count"] = 1
            return merged
        
        # Generate canonical ID from first record (most representative)
        canonical_id = self.generate_canonical_id(records[0])
        
        # Collect all values for each field
        field_values = defaultdict(list)
        
        for i, record in enumerate(records):
            for field_name, value in record.items():
                if value is not None and str(value).strip():
                    field_values[field_name].append({
                        "value": value,
                        "source_record_id": record_ids[i],
                        "source": record.get("source", "unknown")
                    })
        
        # Resolve conflicts for each field
        merged_record = {"canonical_id": canonical_id}
        
        for field_name, values in field_values.items():
            if field_name in ["provider_id", "source"]:
                # Don't merge original provider IDs or sources
                continue
            
            resolved_value = self.resolve_field_conflict(field_name, values)
            merged_record[field_name] = resolved_value
            
            # Store all values if configured
            if self.keep_all_values and len(set(str(v["value"]) for v in values if v["value"])) > 1:
                merged_record[f"{field_name}_all"] = [v["value"] for v in values]
                if self.provenance_tags:
                    merged_record[f"{field_name}_sources"] = [v["source"] for v in values]
        
        # Add merge metadata
        merged_record["merged_from"] = record_ids
        merged_record["merge_count"] = len(records)
        merged_record["original_sources"] = list(set(record.get("source", "unknown") for record in records))
        
        return merged_record
    
    def merge_pairs(self, providers_df: pd.DataFrame, 
                   merge_pairs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge provider pairs based on merge decisions.
        
        Args:
            providers_df: DataFrame with all provider records
            merge_pairs_df: DataFrame with pairs to merge (record_id_1, record_id_2)
            
        Returns:
            Tuple of (merged_providers_df, merge_log_df)
        """
        logger.info(f"Merging {len(merge_pairs_df)} provider pairs")
        
        # Build merge groups (connected components)
        merge_groups = self._build_merge_groups(merge_pairs_df)
        
        merged_records = []
        merge_log = []
        
        for group_id, record_ids in enumerate(merge_groups):
            # Get records for this group
            group_records = []
            for record_id in record_ids:
                if record_id in providers_df.index:
                    group_records.append(providers_df.loc[record_id].to_dict())
            
            if not group_records:
                continue
            
            # Merge records
            merged_record = self.merge_records(group_records, record_ids)
            merged_records.append(merged_record)
            
            # Log merge operation
            log_entry = {
                "merge_group_id": group_id,
                "canonical_id": merged_record["canonical_id"],
                "merged_record_ids": record_ids,
                "merge_count": len(record_ids),
                "original_sources": merged_record["original_sources"]
            }
            merge_log.append(log_entry)
        
        # Create DataFrames
        merged_df = pd.DataFrame(merged_records)
        merged_df.set_index("canonical_id", inplace=True)
        
        log_df = pd.DataFrame(merge_log)
        
        logger.info(f"Merged {len(merge_groups)} groups into {len(merged_df)} canonical records")
        
        return merged_df, log_df
    
    def _build_merge_groups(self, merge_pairs_df: pd.DataFrame) -> List[List[int]]:
        """
        Build merge groups from pairs using connected components algorithm.
        
        Args:
            merge_pairs_df: DataFrame with merge pairs
            
        Returns:
            List of merge groups (each group is a list of record IDs)
        """
        # Build adjacency list
        adjacency = defaultdict(set)
        
        for _, pair in merge_pairs_df.iterrows():
            id1 = pair["record_id_1"]
            id2 = pair["record_id_2"]
            adjacency[id1].add(id2)
            adjacency[id2].add(id1)
        
        # Find connected components
        visited = set()
        merge_groups = []
        
        for record_id in adjacency:
            if record_id not in visited:
                # BFS to find connected component
                group = []
                queue = [record_id]
                
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        group.append(current)
                        queue.extend(adjacency[current] - visited)
                
                merge_groups.append(sorted(group))
        
        return merge_groups
    
    def apply_auto_merges(self, hybrid_df: pd.DataFrame, 
                         providers_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply automatic merges based on hybrid scoring decisions.
        
        Args:
            hybrid_df: DataFrame with hybrid scores and decisions
            providers_df: DataFrame with provider records
            
        Returns:
            Tuple of (merged_providers_df, unmerged_providers_df, merge_log_df)
        """
        # Filter auto-merge pairs
        auto_merge_pairs = hybrid_df[hybrid_df["auto_merge"] == True].copy()
        
        if len(auto_merge_pairs) == 0:
            logger.info("No auto-merge pairs found")
            return providers_df, pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"Auto-merging {len(auto_merge_pairs)} pairs")
        
        # Perform merges
        merged_df, merge_log_df = self.merge_pairs(providers_df, auto_merge_pairs)
        
        # Get unmerged records (those not in any merge group)
        merged_record_ids = set()
        for merged_ids in merge_log_df["merged_record_ids"]:
            merged_record_ids.update(merged_ids)
        
        unmerged_df = providers_df[~providers_df.index.isin(merged_record_ids)]
        
        logger.info(f"Merge completed: {len(merged_df)} merged records, {len(unmerged_df)} unmerged records")
        
        return merged_df, unmerged_df, merge_log_df
    
    def get_merge_statistics(self, merge_log_df: pd.DataFrame, 
                           original_count: int) -> Dict[str, any]:
        """
        Calculate merge statistics.
        
        Args:
            merge_log_df: DataFrame with merge log
            original_count: Original number of records
            
        Returns:
            Dictionary with merge statistics
        """
        if merge_log_df.empty:
            return {
                "original_count": original_count,
                "merged_count": original_count,
                "duplicate_reduction": 0,
                "duplicate_reduction_percentage": 0.0,
                "merge_groups": 0
            }
        
        total_merged_records = merge_log_df["merge_count"].sum()
        canonical_records = len(merge_log_df)
        duplicate_reduction = total_merged_records - canonical_records
        reduction_percentage = (duplicate_reduction / original_count) * 100 if original_count > 0 else 0
        
        # Group size distribution
        group_sizes = merge_log_df["merge_count"].value_counts().to_dict()
        
        statistics = {
            "original_count": original_count,
            "merged_count": original_count - duplicate_reduction,
            "duplicate_reduction": duplicate_reduction,
            "duplicate_reduction_percentage": reduction_percentage,
            "merge_groups": canonical_records,
            "group_size_distribution": group_sizes,
            "avg_group_size": merge_log_df["merge_count"].mean(),
            "max_group_size": merge_log_df["merge_count"].max()
        }
        
        return statistics
    
    def rollback_merge(self, canonical_id: str, providers_df: pd.DataFrame, 
                      merge_log_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rollback a specific merge operation.
        
        Args:
            canonical_id: Canonical ID to rollback
            providers_df: Current providers DataFrame
            merge_log_df: Merge log DataFrame
            
        Returns:
            Updated providers DataFrame with merge rolled back
        """
        # Find merge entry
        merge_entry = merge_log_df[merge_log_df["canonical_id"] == canonical_id]
        
        if merge_entry.empty:
            logger.warning(f"No merge found for canonical ID {canonical_id}")
            return providers_df
        
        # Remove merged record
        result_df = providers_df.drop(canonical_id, errors="ignore")
        
        # Note: In a full implementation, this would restore original records
        # For now, we just log the rollback
        logger.info(f"Rolled back merge for canonical ID {canonical_id}")
        
        return result_df


def merge_provider_records(providers_df: pd.DataFrame,
                          hybrid_df: pd.DataFrame,
                          config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to merge provider records.
    
    Args:
        providers_df: Provider records DataFrame
        hybrid_df: Hybrid scoring results DataFrame
        config: Merge configuration
        
    Returns:
        Tuple of (merged_providers_df, unmerged_providers_df, merge_log_df)
    """
    merger = ProviderMerger(config)
    return merger.apply_auto_merges(hybrid_df, providers_df)
