"""
Schema validation using Great Expectations for ProviderVerify.

Validates provider data against expected schema and data quality rules.
Fails fast on invalid data to ensure data quality and compliance.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import BatchRequest
from great_expectations.dataset.pandas_dataset import PandasDataset
from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult

logger = logging.getLogger(__name__)


class ProviderSchemaValidator:
    """
    Validates provider data schema and quality using Great Expectations.
    
    Security note: Ensure PHI is properly handled in validation logs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator with configuration.
        
        Args:
            config: Configuration dictionary with schema rules
        """
        self.config = config
        self.required_columns = config.get("required_columns", [])
        self.column_types = config.get("column_types", {})
        
        # Initialize Great Expectations context
        self.context = gx.get_context()
        self.data_source_name = "provider_data"
        
        # Create data source if not exists
        self._setup_data_source()
        
        logger.info("Initialized ProviderSchemaValidator")
    
    def _setup_data_source(self):
        """Set up Great Expectations data source."""
        try:
            # Add pandas data source
            datasource_config = {
                "name": self.data_source_name,
                "class_name": "Datasource",
                "module_name": "great_expectations.datasource",
                "execution_engine": {
                    "class_name": "PandasExecutionEngine",
                    "module_name": "great_expectations.execution_engine",
                },
                "data_connectors": {
                    "default_runtime_data_connector": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["default_identifier_name"],
                    },
                },
            }
            
            self.context.test_yaml_config(datasource_config)
            self.context.add_datasource(**datasource_config)
            
        except Exception as e:
            logger.warning(f"Data source may already exist: {e}")
    
    def create_expectations(self) -> gx.ExpectationSuite:
        """
        Create expectation suite for provider data validation.
        
        Returns:
            ExpectationSuite with validation rules
        """
        suite_name = "provider_data_validation"
        
        # Create expectation suite
        suite = self.context.add_expectation_suite(suite_name)
        
        # Add expectations for required columns
        for column in self.required_columns:
            suite.add_expectation(
                gx.expectations.ExpectColumnToExist(column=column)
            )
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
            )
        
        # Add expectations for column types
        for column, expected_type in self.column_types.items():
            if expected_type == "string":
                suite.add_expectation(
                    gx.expectations.ExpectColumnValuesToBeOfType(column=column, type_="str")
                )
        
        # Add data quality expectations
        suite.add_expectation(
            gx.expectations.ExpectTableRowCountToBeBetween(min_value=1, max_value=1000000)
        )
        
        # Provider ID should be unique
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeUnique(column="provider_id")
        )
        
        # Name should not be empty
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="raw_name")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValueLengthsToBeBetween(
                column="raw_name", min_value=2, max_value=100
            )
        )
        
        # Email format validation
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToMatchRegex(
                column="email",
                regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            )
        )
        
        # Phone number format validation (basic)
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToMatchRegex(
                column="phone",
                regex=r"^[\d\-\(\)\s\+]+$"
            )
        )
        
        # Source should be one of expected values
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="source",
                value_set=["EHR", "HMO", "Medicaid", "Medicare", "Public Registry", "Insurance"]
            )
        )
        
        logger.info(f"Created expectation suite '{suite_name}' with {len(suite.expectations)} expectations")
        return suite
    
    def validate_data(self, df: pd.DataFrame, 
                     expectation_suite: Optional[gx.ExpectationSuite] = None) -> CheckpointResult:
        """
        Validate DataFrame against expectation suite.
        
        Args:
            df: DataFrame to validate
            expectation_suite: Custom expectation suite (optional)
            
        Returns:
            CheckpointResult with validation results
        """
        try:
            # Create expectation suite if not provided
            if expectation_suite is None:
                expectation_suite = self.create_expectations()
            
            # Create batch request
            batch_request = BatchRequest(
                datasource_name=self.data_source_name,
                data_connector_name="default_runtime_data_connector",
                data_asset_name="provider_data",
                runtime_parameters={"batch_data": df},
                batch_identifiers={"default_identifier_name": "provider_batch"},
            )
            
            # Create checkpoint
            checkpoint_config = {
                "name": "provider_validation_checkpoint",
                "config_version": 1.0,
                "class_name": "Checkpoint",
                "validations": [
                    {
                        "batch_request": batch_request,
                        "expectation_suite_name": expectation_suite.expectation_suite_name,
                    }
                ],
            }
            
            checkpoint = self.context.add_or_update_checkpoint(**checkpoint_config)
            
            # Run validation
            checkpoint_result = checkpoint.run()
            
            # Log validation summary
            success_count = sum(1 for result in checkpoint_result.run_results.values() 
                              if result["validation_result"]["success"])
            total_count = len(checkpoint_result.run_results)
            
            logger.info(f"Validation completed: {success_count}/{total_count} expectations passed")
            
            # Log failed expectations (without PHI)
            for run_result in checkpoint_result.run_results.values():
                validation_result = run_result["validation_result"]
                if not validation_result["success"]:
                    for expectation_result in validation_result["results"]:
                        if not expectation_result["success"]:
                            logger.warning(
                                f"Failed expectation: {expectation_result['expectation_config']['expectation_type']} "
                                f"for column: {expectation_result['expectation_config'].get('kwargs', {}).get('column', 'N/A')}"
                            )
            
            return checkpoint_result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def get_validation_summary(self, checkpoint_result: CheckpointResult) -> Dict[str, Any]:
        """
        Extract validation summary from checkpoint result.
        
        Args:
            checkpoint_result: Result from validation checkpoint
            
        Returns:
            Dictionary with validation summary
        """
        summary = {
            "success": True,
            "total_expectations": 0,
            "successful_expectations": 0,
            "failed_expectations": 0,
            "validation_statistics": {},
            "failed_expectations_details": []
        }
        
        for run_result in checkpoint_result.run_results.values():
            validation_result = run_result["validation_result"]
            
            summary["total_expectations"] += len(validation_result["results"])
            
            for expectation_result in validation_result["results"]:
                if expectation_result["success"]:
                    summary["successful_expectations"] += 1
                else:
                    summary["failed_expectations"] += 1
                    summary["success"] = False
                    
                    # Add failed expectation details (without PHI)
                    failed_detail = {
                        "expectation_type": expectation_result["expectation_config"]["expectation_type"],
                        "column": expectation_result["expectation_config"].get("kwargs", {}).get("column", "N/A"),
                        "result": expectation_result["result"]
                    }
                    summary["failed_expectations_details"].append(failed_detail)
        
        # Calculate success rate
        if summary["total_expectations"] > 0:
            summary["success_rate"] = summary["successful_expectations"] / summary["total_expectations"]
        else:
            summary["success_rate"] = 0.0
        
        logger.info(f"Validation summary: {summary['successful_expectations']}/{summary['total_expectations']} passed "
                   f"({summary['success_rate']:.2%} success rate)")
        
        return summary
    
    def clean_invalid_data(self, df: pd.DataFrame, 
                          checkpoint_result: CheckpointResult) -> pd.DataFrame:
        """
        Clean invalid data based on validation results.
        
        Args:
            df: Original DataFrame
            checkpoint_result: Validation results
            
        Returns:
            Cleaned DataFrame
        """
        original_rows = len(df)
        cleaned_df = df.copy()
        
        # Process failed expectations
        for run_result in checkpoint_result.run_results.values():
            validation_result = run_result["validation_result"]
            
            for expectation_result in validation_result["results"]:
                if not expectation_result["success"]:
                    expectation_type = expectation_result["expectation_config"]["expectation_type"]
                    column = expectation_result["expectation_config"].get("kwargs", {}).get("column")
                    
                    if expectation_type == "expect_column_values_to_not_be_null" and column:
                        # Remove rows with null values in required columns
                        null_mask = cleaned_df[column].isnull()
                        removed_count = null_mask.sum()
                        cleaned_df = cleaned_df[~null_mask]
                        logger.info(f"Removed {removed_count} rows with null values in column '{column}'")
                    
                    elif expectation_type == "expect_column_values_to_be_unique" and column == "provider_id":
                        # Remove duplicate provider IDs, keeping first occurrence
                        duplicate_mask = cleaned_df[column].duplicated(keep='first')
                        removed_count = duplicate_mask.sum()
                        cleaned_df = cleaned_df[~duplicate_mask]
                        logger.info(f"Removed {removed_count} duplicate provider IDs")
        
        cleaned_rows = len(cleaned_df)
        removed_rows = original_rows - cleaned_rows
        
        logger.info(f"Data cleaning completed: removed {removed_rows} invalid rows "
                   f"({removed_rows/original_rows:.2%} of data)")
        
        return cleaned_df


def validate_provider_data(df: pd.DataFrame, config: Dict[str, Any]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to validate and clean provider data.
    
    Args:
        df: Provider DataFrame to validate
        config: Validation configuration
        
    Returns:
        Tuple of (cleaned_df, validation_summary)
    """
    validator = ProviderSchemaValidator(config)
    checkpoint_result = validator.validate_data(df)
    summary = validator.get_validation_summary(checkpoint_result)
    
    if summary["success"]:
        logger.info("All validation expectations passed")
        return df, summary
    else:
        logger.warning("Validation failed, cleaning invalid data")
        cleaned_df = validator.clean_invalid_data(df, checkpoint_result)
        return cleaned_df, summary
