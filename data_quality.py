"""
Automated Data Quality Check Framework

This module provides a flexible, extensible framework for defining,
executing, and reporting on data quality checks across datasets.

The framework follows a rule-based approach where data quality checks
are defined as rules that can be evaluated against datasets to produce
validation results and metrics.

Business Value:
- Early detection of data quality issues before they impact business decisions
- Consistent evaluation of data across systems and processes
- Reduced time spent on manual quality reviews
- Enhanced trust in data through documented quality assessments
- Support for data governance and compliance requirements
"""

import os
import pandas as pd
import numpy as np
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Any, Callable, Union, Optional, Tuple
import re
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from email.message import EmailMessage
import smtplib
import importlib
import inspect


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_quality.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataQualityRule:
    """
    Base class for data quality rules that can be evaluated against datasets.
    
    A DataQualityRule defines a specific check that can be performed on data
    to assess its quality. This could include checks for completeness,
    validity, consistency, timeliness, or other quality dimensions.
    """
    
    def __init__(self, rule_id: str, description: str, severity: str = "medium"):
        """
        Initialize a data quality rule.
        
        Args:
            rule_id: Unique identifier for the rule
            description: Human-readable description of what the rule checks
            severity: Impact level if rule fails ('low', 'medium', 'high', 'critical')
        """
        self.rule_id = rule_id
        self.description = description
        self.severity = severity.lower()
        # Validate severity
        valid_severities = ["low", "medium", "high", "critical"]
        if self.severity not in valid_severities:
            logger.warning(f"Invalid severity '{severity}' for rule {rule_id}, "
                         f"using 'medium' instead. Valid options: {valid_severities}")
            self.severity = "medium"
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the rule against a dataset.
        
        Args:
            data: The dataset to check
            **kwargs: Additional parameters for rule evaluation
            
        Returns:
            A dictionary containing evaluation results with at least these keys:
            - success: Boolean indicating if the check passed
            - message: Description of the result
            - metrics: Dictionary of quantitative metrics from the check
        """
        # This is a base method that should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the rule to a dictionary representation.
        
        Returns:
            Dictionary representation of the rule
        """
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "severity": self.severity,
            "type": self.__class__.__name__
        }


class CompletenessRule(DataQualityRule):
    """
    Rule to check for completeness (absence of null values) in specified columns.
    """
    
    def __init__(self, rule_id: str, columns: List[str], 
                threshold: float = 1.0, description: str = None, 
                severity: str = "medium"):
        """
        Initialize a completeness rule.
        
        Args:
            rule_id: Unique identifier for the rule
            columns: List of column names to check for completeness
            threshold: Minimum acceptable completion rate (0.0 to 1.0)
            description: Human-readable description (auto-generated if None)
            severity: Impact level if rule fails
        """
        if description is None:
            column_str = ", ".join(columns) if len(columns) <= 3 else f"{len(columns)} columns"
            threshold_pct = threshold * 100
            description = f"Check that {column_str} has at least {threshold_pct:.1f}% non-null values"
        
        super().__init__(rule_id, description, severity)
        self.columns = columns
        self.threshold = threshold
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Evaluate completeness in specified columns.
        
        Args:
            data: The dataset to check
            **kwargs: Additional parameters (unused)
            
        Returns:
            Evaluation results
        """
        # Verify columns exist in the dataframe
        missing_columns = [col for col in self.columns if col not in data.columns]
        if missing_columns:
            return {
                "success": False,
                "message": f"Columns not found in dataframe: {', '.join(missing_columns)}",
                "metrics": {"missing_columns_count": len(missing_columns)},
                "details": {"missing_columns": missing_columns}
            }
        
        # Calculate completion rate for each column
        completion_rates = {}
        failing_columns = []
        
        for column in self.columns:
            total_count = len(data)
            null_count = data[column].isnull().sum()
            completion_rate = (total_count - null_count) / total_count if total_count > 0 else 0.0
            completion_rates[column] = completion_rate
            
            if completion_rate < self.threshold:
                failing_columns.append(column)
        
        # Determine overall success
        success = len(failing_columns) == 0
        
        # Create result
        if success:
            message = "All columns meet completeness threshold"
        else:
            failing_str = ", ".join(failing_columns)
            message = f"Columns below completeness threshold: {failing_str}"
        
        return {
            "success": success,
            "message": message,
            "metrics": {
                "columns_checked": len(self.columns),
                "columns_failed": len(failing_columns),
                "completion_rates": completion_rates,
                "average_completion_rate": sum(completion_rates.values()) / len(completion_rates) 
                                        if completion_rates else 0.0
            },
            "details": {
                "failing_columns": failing_columns,
                "threshold": self.threshold
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the rule to a dictionary representation.
        
        Returns:
            Dictionary representation of the rule
        """
        rule_dict = super().to_dict()
        rule_dict.update({
            "columns": self.columns,
            "threshold": self.threshold
        })
        return rule_dict


class UniquenessRule(DataQualityRule):
    """
    Rule to check for uniqueness in specified columns (no duplicate values).
    """
    
    def __init__(self, rule_id: str, columns: Union[List[str], str], 
                allow_nulls: bool = False, threshold: float = 1.0,
                description: str = None, severity: str = "medium"):
        """
        Initialize a uniqueness rule.
        
        Args:
            rule_id: Unique identifier for the rule
            columns: Column name or list of column names to check for uniqueness
            allow_nulls: Whether to allow null values in uniqueness check
            threshold: Minimum acceptable uniqueness rate (0.0 to 1.0)
            description: Human-readable description (auto-generated if None)
            severity: Impact level if rule fails
        """
        # Convert single column to list for consistent handling
        self.columns = [columns] if isinstance(columns, str) else columns
        
        if description is None:
            column_str = ", ".join(self.columns) if len(self.columns) <= 3 else f"{len(self.columns)} columns"
            threshold_pct = threshold * 100
            description = f"Check that {column_str} contains at least {threshold_pct:.1f}% unique values"
        
        super().__init__(rule_id, description, severity)
        self.allow_nulls = allow_nulls
        self.threshold = threshold
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Evaluate uniqueness in specified columns.
        
        Args:
            data: The dataset to check
            **kwargs: Additional parameters (unused)
            
        Returns:
            Evaluation results
        """
        # Verify columns exist
        missing_columns = [col for col in self.columns if col not in data.columns]
        if missing_columns:
            return {
                "success": False,
                "message": f"Columns not found in dataframe: {', '.join(missing_columns)}",
                "metrics": {"missing_columns_count": len(missing_columns)},
                "details": {"missing_columns": missing_columns}
            }
        
        # Prepare data for uniqueness check
        check_data = data[self.columns].copy()
        
        # Handle nulls according to rule configuration
        if not self.allow_nulls:
            # If nulls not allowed, remove rows with any nulls
            check_data = check_data.dropna()
        
        # Calculate uniqueness
        total_rows = len(check_data)
        unique_rows = len(check_data.drop_duplicates())
        
        # Handle empty dataset case
        if total_rows == 0:
            return {
                "success": False,
                "message": "No data to check uniqueness (empty dataset after null handling)",
                "metrics": {
                    "uniqueness_rate": 0.0,
                    "duplicate_count": 0
                },
                "details": {
                    "total_rows": total_rows,
                    "unique_rows": unique_rows,
                    "threshold": self.threshold
                }
            }
        
        uniqueness_rate = unique_rows / total_rows
        duplicate_count = total_rows - unique_rows
        
        # Determine success
        success = uniqueness_rate >= self.threshold
        
        # Create message
        if success:
            message = f"Uniqueness check passed with rate {uniqueness_rate:.2%}"
        else:
            message = (f"Uniqueness check failed with rate {uniqueness_rate:.2%} "
                      f"below threshold {self.threshold:.2%}")
        
        # Get examples of duplicates for details
        duplicate_examples = []
        if duplicate_count > 0:
            # Find duplicated values
            duplicated_mask = check_data.duplicated(keep=False)
            duplicated_rows = check_data[duplicated_mask]
            
            # Get the top 5 most common duplicates
            duplicate_examples = (duplicated_rows.value_counts()
                                .head(5)
                                .reset_index()
                                .to_dict(orient='records'))
        
        return {
            "success": success,
            "message": message,
            "metrics": {
                "uniqueness_rate": uniqueness_rate,
                "duplicate_count": duplicate_count
            },
            "details": {
                "total_rows": total_rows,
                "unique_rows": unique_rows,
                "threshold": self.threshold,
                "duplicate_examples": duplicate_examples
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the rule to a dictionary representation.
        
        Returns:
            Dictionary representation of the rule
        """
        rule_dict = super().to_dict()
        rule_dict.update({
            "columns": self.columns,
            "allow_nulls": self.allow_nulls,
            "threshold": self.threshold
        })
        return rule_dict


class ValueRangeRule(DataQualityRule):
    """
    Rule to check if values in a column fall within an expected range.
    """
    
    def __init__(self, rule_id: str, column: str, min_value: float = None, 
                max_value: float = None, include_min: bool = True, 
                include_max: bool = True, description: str = None, 
                severity: str = "medium"):
        """
        Initialize a value range rule.
        
        Args:
            rule_id: Unique identifier for the rule
            column: Column name to check
            min_value: Minimum acceptable value (None for no lower bound)
            max_value: Maximum acceptable value (None for no upper bound)
            include_min: Whether minimum value is included in valid range
            include_max: Whether maximum value is included in valid range
            description: Human-readable description (auto-generated if None)
            severity: Impact level if rule fails
        """
        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified")
        
        if description is None:
            # Build range description based on provided bounds
            bounds = []
            if min_value is not None:
                bounds.append(f"{'≥' if include_min else '>'} {min_value}")
            if max_value is not None:
                bounds.append(f"{'≤' if include_max else '<'} {max_value}")
            
            range_str = " and ".join(bounds)
            description = f"Check that values in column '{column}' are {range_str}"
        
        super().__init__(rule_id, description, severity)
        self.column = column
        self.min_value = min_value
        self.max_value = max_value
        self.include_min = include_min
        self.include_max = include_max
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Evaluate if values fall within the specified range.
        
        Args:
            data: The dataset to check
            **kwargs: Additional parameters (unused)
            
        Returns:
            Evaluation results
        """
        # Verify column exists
        if self.column not in data.columns:
            return {
                "success": False,
                "message": f"Column '{self.column}' not found in dataframe",
                "metrics": {},
                "details": {"missing_column": self.column}
            }
        
        # Filter out nulls
        column_data = data[self.column].dropna()
        
        # Handle empty column case
        if len(column_data) == 0:
            return {
                "success": False,
                "message": f"No data to check range (column '{self.column}' is empty or all null)",
                "metrics": {
                    "in_range_rate": 0.0,
                    "out_of_range_count": 0
                },
                "details": {
                    "total_values": 0,
                    "null_values": len(data) - len(column_data)
                }
            }
        
        # Check lower bound if specified
        if self.min_value is not None:
            if self.include_min:
                min_mask = column_data >= self.min_value
            else:
                min_mask = column_data > self.min_value
        else:
            min_mask = pd.Series(True, index=column_data.index)
        
        # Check upper bound if specified
        if self.max_value is not None:
            if self.include_max:
                max_mask = column_data <= self.max_value
            else:
                max_mask = column_data < self.max_value
        else:
            max_mask = pd.Series(True, index=column_data.index)
        
        # Combine masks to find values in range
        in_range_mask = min_mask & max_mask
        
        # Calculate metrics
        total_values = len(column_data)
        in_range_count = in_range_mask.sum()
        out_of_range_count = total_values - in_range_count
        in_range_rate = in_range_count / total_values
        
        # Determine success (all values in range)
        success = out_of_range_count == 0
        
        # Create message
        if success:
            message = f"All values in column '{self.column}' are within the specified range"
        else:
            message = (f"{out_of_range_count} values in column '{self.column}' "
                      f"are outside the specified range ({in_range_rate:.2%} in range)")
        
        # Get statistics on out-of-range values
        out_of_range_stats = {}
        if out_of_range_count > 0:
            out_of_range_values = column_data[~in_range_mask]
            out_of_range_stats = {
                "min": float(out_of_range_values.min()),
                "max": float(out_of_range_values.max()),
                "mean": float(out_of_range_values.mean()),
                "median": float(out_of_range_values.median()),
                "examples": out_of_range_values.head(5).tolist()
            }
        
        return {
            "success": success,
            "message": message,
            "metrics": {
                "in_range_rate": in_range_rate,
                "out_of_range_count": int(out_of_range_count)
            },
            "details": {
                "total_values": total_values,
                "non_matching_examples": out_of_range_stats,
                "reference_source": str(self.reference_data) if isinstance(self.reference_data, str) else "dataframe",
                "reference_column": self.reference_column,
                "threshold": self.threshold
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the rule to a dictionary representation.
        
        Returns:
            Dictionary representation of the rule
        """
        rule_dict = super().to_dict()
        rule_dict.update({
            "column": self.column,
            "reference_data": str(self.reference_data),
            "reference_column": self.reference_column,
            "ignore_case": self.ignore_case,
            "threshold": self.threshold
        })
        return rule_dict


class StatisticalRule(DataQualityRule):
    """
    Rule to check if statistical properties of a column are within expected ranges.
    """
    
    def __init__(self, rule_id: str, column: str, stat_type: str,
                expected_value: float, tolerance: float,
                description: str = None, severity: str = "medium"):
        """
        Initialize a statistical rule.
        
        Args:
            rule_id: Unique identifier for the rule
            column: Column name to check
            stat_type: Type of statistic to check ('mean', 'median', 'std', 'min', 'max', 'sum')
            expected_value: Expected value of the statistic
            tolerance: Allowed deviation from expected value (absolute or percentage based on tolerance_type)
            description: Human-readable description (auto-generated if None)
            severity: Impact level if rule fails
        """
        valid_stat_types = ['mean', 'median', 'std', 'min', 'max', 'sum']
        if stat_type not in valid_stat_types:
            raise ValueError(f"Invalid stat_type '{stat_type}'. Valid options: {valid_stat_types}")
        
        if description is None:
            description = (f"Check that the {stat_type} of column '{column}' "
                          f"is within {tolerance} of expected value {expected_value}")
        
        super().__init__(rule_id, description, severity)
        self.column = column
        self.stat_type = stat_type
        self.expected_value = expected_value
        self.tolerance = tolerance
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Evaluate if statistical properties are within expected ranges.
        
        Args:
            data: The dataset to check
            **kwargs: Additional parameters (unused)
            
        Returns:
            Evaluation results
        """
        # Verify column exists
        if self.column not in data.columns:
            return {
                "success": False,
                "message": f"Column '{self.column}' not found in dataframe",
                "metrics": {},
                "details": {"missing_column": self.column}
            }
        
        # Get numeric data (drop nulls)
        column_data = pd.to_numeric(data[self.column], errors='coerce').dropna()
        
        # Handle empty column case
        if len(column_data) == 0:
            return {
                "success": False,
                "message": f"No numeric data to analyze in column '{self.column}'",
                "metrics": {},
                "details": {
                    "null_count": data[self.column].isnull().sum(),
                    "non_numeric_count": len(data) - data[self.column].isnull().sum() - len(column_data)
                }
            }
        
        # Calculate the requested statistic
        if self.stat_type == 'mean':
            actual_value = column_data.mean()
        elif self.stat_type == 'median':
            actual_value = column_data.median()
        elif self.stat_type == 'std':
            actual_value = column_data.std()
        elif self.stat_type == 'min':
            actual_value = column_data.min()
        elif self.stat_type == 'max':
            actual_value = column_data.max()
        else:  # sum
            actual_value = column_data.sum()
        
        # Calculate acceptable range
        lower_bound = self.expected_value - self.tolerance
        upper_bound = self.expected_value + self.tolerance
        
        # Check if actual value is within range
        in_range = lower_bound <= actual_value <= upper_bound
        
        # Calculate the deviation
        deviation = actual_value - self.expected_value
        deviation_percentage = (deviation / self.expected_value) * 100 if self.expected_value != 0 else float('inf')
        
        # Determine success
        success = in_range
        
        # Create message
        if success:
            message = (f"Statistical check passed: {self.stat_type} of column '{self.column}' "
                      f"is {actual_value:.4g}, within tolerance of expected value {self.expected_value}")
        else:
            message = (f"Statistical check failed: {self.stat_type} of column '{self.column}' "
                      f"is {actual_value:.4g}, outside tolerance range "
                      f"[{lower_bound:.4g}, {upper_bound:.4g}]")
        
        return {
            "success": success,
            "message": message,
            "metrics": {
                "actual_value": float(actual_value),
                "deviation": float(deviation),
                "deviation_percentage": float(deviation_percentage)
            },
            "details": {
                "expected_value": self.expected_value,
                "tolerance": self.tolerance,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "data_count": len(column_data),
                "data_summary": {
                    "mean": float(column_data.mean()),
                    "median": float(column_data.median()),
                    "std": float(column_data.std()),
                    "min": float(column_data.min()),
                    "max": float(column_data.max())
                }
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the rule to a dictionary representation.
        
        Returns:
            Dictionary representation of the rule
        """
        rule_dict = super().to_dict()
        rule_dict.update({
            "column": self.column,
            "stat_type": self.stat_type,
            "expected_value": self.expected_value,
            "tolerance": self.tolerance
        })
        return rule_dict


class CustomSQLRule(DataQualityRule):
    """
    Rule to check data quality using a custom SQL query.
    """
    
    def __init__(self, rule_id: str, sql_query: str, 
                validation_type: str = "boolean", expected_value: Any = True,
                description: str = None, severity: str = "medium"):
        """
        Initialize a custom SQL rule.
        
        Args:
            rule_id: Unique identifier for the rule
            sql_query: SQL query to execute for validation
            validation_type: How to validate the result ('boolean', 'equals', 'range')
            expected_value: Expected value or range for validation
            description: Human-readable description (auto-generated if None)
            severity: Impact level if rule fails
        """
        if description is None:
            # Truncate SQL for description
            max_sql_length = 50
            truncated_sql = (sql_query[:max_sql_length] + '...'
                           if len(sql_query) > max_sql_length else sql_query)
            description = f"Check data quality using custom SQL: {truncated_sql}"
        
        super().__init__(rule_id, description, severity)
        self.sql_query = sql_query
        self.validation_type = validation_type
        self.expected_value = expected_value
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Evaluate using custom SQL.
        
        Args:
            data: Ignored for SQL rules (uses database connection)
            **kwargs: Must include 'connection' for database access
            
        Returns:
            Evaluation results
        """
        # Get database connection
        connection = kwargs.get("connection")
        if connection is None:
            return {
                "success": False,
                "message": "Database connection required for SQL rule",
                "metrics": {},
                "details": {"sql_query": self.sql_query}
            }
        
        try:
            # Execute query
            result = pd.read_sql(self.sql_query, connection)
            
            # Handle empty result case
            if len(result) == 0:
                return {
                    "success": False,
                    "message": "SQL query returned no results",
                    "metrics": {},
                    "details": {"sql_query": self.sql_query}
                }
            
            # Extract validation value (assumes first column, first row)
            actual_value = result.iloc[0, 0]
            
            # Validate based on validation type
            if self.validation_type == 'boolean':
                # Convert to boolean if needed
                if isinstance(actual_value, (int, float)):
                    success = bool(actual_value)
                else:
                    success = actual_value in (True, 'true', 'True', 'TRUE', 'yes', 'Yes', 'YES', '1', 't', 'T')
                
                message = f"SQL validation {'passed' if success else 'failed'} with result: {actual_value}"
            
            elif self.validation_type == 'equals':
                success = actual_value == self.expected_value
                message = (f"SQL validation {'passed' if success else 'failed'}: "
                          f"expected {self.expected_value}, got {actual_value}")
            
            elif self.validation_type == 'range':
                # Expected value should be a tuple/list with (min, max)
                if not isinstance(self.expected_value, (list, tuple)) or len(self.expected_value) != 2:
                    return {
                        "success": False,
                        "message": "Invalid expected_value for range validation: must be (min, max) tuple",
                        "metrics": {"actual_value": actual_value},
                        "details": {"sql_query": self.sql_query}
                    }
                
                min_val, max_val = self.expected_value
                success = min_val <= actual_value <= max_val
                message = (f"SQL validation {'passed' if success else 'failed'}: "
                          f"expected range [{min_val}, {max_val}], got {actual_value}")
            
            else:
                return {
                    "success": False,
                    "message": f"Invalid validation_type: {self.validation_type}",
                    "metrics": {"actual_value": actual_value},
                    "details": {"sql_query": self.sql_query}
                }
            
            return {
                "success": success,
                "message": message,
                "metrics": {"actual_value": actual_value},
                "details": {
                    "sql_query": self.sql_query,
                    "validation_type": self.validation_type,
                    "expected_value": self.expected_value,
                    "full_result": result.to_dict(orient='records') if len(result) < 10 else 
                                  f"{len(result)} rows returned"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error executing SQL query: {str(e)}",
                "metrics": {},
                "details": {"sql_query": self.sql_query, "error": str(e)}
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the rule to a dictionary representation.
        
        Returns:
            Dictionary representation of the rule
        """
        rule_dict = super().to_dict()
        rule_dict.update({
            "sql_query": self.sql_query,
            "validation_type": self.validation_type,
            "expected_value": self.expected_value
        })
        return rule_dict


class CustomPythonRule(DataQualityRule):
    """
    Rule to check data quality using a custom Python function.
    """
    
    def __init__(self, rule_id: str, check_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, Any]],
                function_args: Dict[str, Any] = None, description: str = None,
                severity: str = "medium"):
        """
        Initialize a custom Python rule.
        
        Args:
            rule_id: Unique identifier for the rule
            check_function: Python function to execute for validation
                           Must take (df, **kwargs) and return dict with at least 'success' key
            function_args: Arguments to pass to the check function
            description: Human-readable description (auto-generated if None)
            severity: Impact level if rule fails
        """
        if description is None:
            func_name = check_function.__name__
            description = f"Check data quality using custom Python function: {func_name}"
        
        super().__init__(rule_id, description, severity)
        self.check_function = check_function
        self.function_args = function_args or {}
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Evaluate using custom Python function.
        
        Args:
            data: The dataset to check
            **kwargs: Additional parameters to pass to check function
            
        Returns:
            Evaluation results
        """
        try:
            # Combine function_args with kwargs (kwargs take precedence)
            combined_args = {**self.function_args, **kwargs}
            
            # Call the check function
            result = self.check_function(data, **combined_args)
            
            # Verify result has required keys
            if not isinstance(result, dict) or 'success' not in result:
                return {
                    "success": False,
                    "message": "Invalid result from check function: must be dict with 'success' key",
                    "metrics": {},
                    "details": {"function": self.check_function.__name__}
                }
            
            # Add default keys if not present
            if 'message' not in result:
                result['message'] = f"Custom check {'passed' if result['success'] else 'failed'}"
            
            if 'metrics' not in result:
                result['metrics'] = {}
            
            if 'details' not in result:
                result['details'] = {"function": self.check_function.__name__}
            else:
                result['details']['function'] = self.check_function.__name__
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error executing custom check function: {str(e)}",
                "metrics": {},
                "details": {"function": self.check_function.__name__, "error": str(e)}
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the rule to a dictionary representation.
        
        Returns:
            Dictionary representation of the rule
        """
        rule_dict = super().to_dict()
        rule_dict.update({
            "function": self.check_function.__name__,
            "function_args": self.function_args
        })
        return rule_dict


class DataQualityValidation:
    """
    Class to run a set of data quality rules and collect results.
    """
    
    def __init__(self, rules: List[DataQualityRule] = None):
        """
        Initialize data quality validation.
        
        Args:
            rules: List of rules to validate (can be added later)
        """
        self.rules = rules or []
        self.results = []
        self.validation_id = None
        self.executed = False
        self.start_time = None
        self.end_time = None
    
    def add_rule(self, rule: DataQualityRule) -> None:
        """
        Add a rule to the validation.
        
        Args:
            rule: Rule to add
        """
        self.rules.append(rule)
    
    def execute(self, data: pd.DataFrame, validation_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Execute all rules on the dataset.
        
        Args:
            data: The dataset to validate
            validation_id: Optional identifier for this validation run
            **kwargs: Additional parameters to pass to rules
            
        Returns:
            A dictionary with validation summary and results
        """
        # Generate validation ID if not provided
        if validation_id is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()[:8]
            validation_id = f"validation-{timestamp}-{data_hash}"
        
        self.validation_id = validation_id
        self.start_time = datetime.now()
        self.results = []
        
        # Track overall metrics
        total_rules = len(self.rules)
        passed_rules = 0
        failed_rules = 0
        
        # Execute each rule
        for rule in self.rules:
            logger.info(f"Executing rule {rule.rule_id}")
            
            try:
                # Evaluate the rule
                result = rule.evaluate(data, **kwargs)
                
                # Add rule metadata to result
                result["rule_id"] = rule.rule_id
                result["description"] = rule.description
                result["severity"] = rule.severity
                
                # Add result to collection
                self.results.append(result)
                
                # Update metrics
                if result["success"]:
                    passed_rules += 1
                else:
                    failed_rules += 1
                    logger.warning(f"Rule {rule.rule_id} failed: {result['message']}")
                
            except Exception as e:
                # Handle rule execution errors
                error_result = {
                    "rule_id": rule.rule_id,
                    "description": rule.description,
                    "severity": rule.severity,
                    "success": False,
                    "message": f"Error executing rule: {str(e)}",
                    "metrics": {},
                    "details": {"error": str(e)}
                }
                self.results.append(error_result)
                failed_rules += 1
                logger.error(f"Error executing rule {rule.rule_id}: {str(e)}", exc_info=True)
        
        self.end_time = datetime.now()
        self.executed = True
        
        # Compute overall metrics
        duration_seconds = (self.end_time - self.start_time).total_seconds()
        critical_failures = sum(1 for r in self.results 
                                if not r["success"] and r["severity"] == "critical")
        high_failures = sum(1 for r in self.results 
                           if not r["success"] and r["severity"] == "high")
        
        # Generate summary
        summary = {
            "validation_id": self.validation_id,
            "timestamp": self.start_time.isoformat(),
            "duration_seconds": duration_seconds,
            "total_rules": total_rules,
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "pass_rate": passed_rules / total_rules if total_rules > 0 else 0,
            "critical_failures": critical_failures,
            "high_failures": high_failures,
            "overall_status": "PASS" if failed_rules == 0 else "FAIL"
        }
        
        # Create validation report
        validation_report = {
            "summary": summary,
            "results": self.results
        }
        
        return validation_report
    
    def save_report(self, output_path: str, format: str = "json") -> str:
        """
        Save validation results to a file.
        
        Args:
            output_path: Path to save the report
            format: Report format ("json" or "yaml")
            
        Returns:
            Path to the saved report
        """
        if not self.executed:
            raise ValueError("Cannot save report: validation has not been executed yet")
        
        # Create validation report
        validation_report = {
            "validation_id": self.validation_id,
            "timestamp": self.start_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "summary": {
                "total_rules": len(self.rules),
                "passed_rules": sum(1 for r in self.results if r["success"]),
                "failed_rules": sum(1 for r in self.results if not r["success"]),
            },
            "results": self.results
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save the report
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(validation_report, f, indent=2)
        elif format.lower() == "yaml":
            with open(output_path, 'w') as f:
                yaml.dump(validation_report, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")
        
        logger.info(f"Saved validation report to {output_path}")
        return output_path
    
    def generate_dashboard(self, output_path: str = None) -> None:
        """
        Generate a visual dashboard of validation results.
        
        Args:
            output_path: Path to save the dashboard HTML
        """
        if not self.executed:
            raise ValueError("Cannot generate dashboard: validation has not been executed yet")
        
        # Convert results to dataframe for easier visualization
        results_df = pd.DataFrame([{
            "rule_id": r["rule_id"],
            "description": r["description"],
            "severity": r["severity"],
            "success": r["success"],
            "message": r["message"]
        } for r in self.results])
        
        # Set up the figure with subplots
        plt.figure(figsize=(12, 10))
        
        # 1. Overall pass rate pie chart
        plt.subplot(2, 2, 1)
        pass_count = sum(results_df["success"])
        fail_count = len(results_df) - pass_count
        plt.pie([pass_count, fail_count], 
                labels=["Pass", "Fail"],
                colors=["#28a745", "#dc3545"],
                autopct='%1.1f%%',
                startangle=90)
        plt.title("Overall Pass Rate")
        
        # 2. Results by severity
        plt.subplot(2, 2, 2)
        severity_results = results_df.groupby(["severity", "success"]).size().unstack(fill_value=0)
        
        # Ensure all columns exist
        for col in [True, False]:
            if col not in severity_results.columns:
                severity_results[col] = 0
        
        # Sort by severity levels
        severity_order = ["critical", "high", "medium", "low"]
        severity_results = severity_results.reindex(severity_order, fill_value=0)
        
        severity_results.plot(kind="barh", color=["#28a745", "#dc3545"], ax=plt.gca())
        plt.title("Results by Severity")
        plt.xlabel("Count")
        plt.ylabel("Severity")
        plt.legend(["Pass", "Fail"])
        
        # 3. Rule results table
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        # Create a table with results
        table_data = []
        for _, row in results_df.iterrows():
            status = "✅" if row["success"] else "❌"
            table_data.append([
                row["rule_id"],
                status,
                row["severity"],
                row["message"][:50] + ('...' if len(row["message"]) > 50 else '')
            ])
        
        table = plt.table(
            cellText=table_data,
            colLabels=["Rule ID", "Status", "Severity", "Message"],
            loc="center",
            cellLoc="left",
            colColours=["#f8f9fa"] * 4,
            colWidths=[0.15, 0.1, 0.15, 0.6]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        plt.title("Rule Results", pad=30)
        
        # Add a title for the entire dashboard
        plt.suptitle(
            f"Data Quality Validation Report: {self.validation_id}\n"
            f"Generated: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=16, y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the dashboard
        if output_path:
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            logger.info(f"Saved validation dashboard to {output_path}")
        else:
            plt.show()
    
    def send_notification(self, 
                         email_to: List[str],
                         email_from: str,
                         smtp_server: str,
                         smtp_port: int = 587,
                         username: str = None,
                         password: str = None,
                         subject_prefix: str = "Data Quality Report",
                         only_on_failures: bool = False,
                         include_details: bool = True) -> bool:
        """
        Send notification email with validation results.
        
        Args:
            email_to: List of recipient email addresses
            email_from: Sender email address
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP authentication username
            password: SMTP authentication password
            subject_prefix: Email subject prefix
            only_on_failures: Only send notification if validation failed
            include_details: Include detailed results in email
            
        Returns:
            Boolean indicating if email was sent successfully
        """
        if not self.executed:
            raise ValueError("Cannot send notification: validation has not been executed yet")
        
        # Check if we should send notification
        has_failures = any(not r["success"] for r in self.results)
        if only_on_failures and not has_failures:
            logger.info("No failures detected, skipping notification")
            return False
        
        # Prepare email
        msg = EmailMessage()
        
        # Set email subject
        status = "FAIL" if has_failures else "PASS"
        subject = f"{subject_prefix}: {status} - {self.validation_id}"
        msg['Subject'] = subject
        msg['From'] = email_from
        msg['To'] = ', '.join(email_to)
        
        # Create email body with HTML formatting
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h2>Data Quality Validation Report</h2>
            <p><strong>Validation ID:</strong> {self.validation_id}</p>
            <p><strong>Timestamp:</strong> {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Status:</strong> <span class="{'pass' if not has_failures else 'fail'}">{status}</span></p>
            
            <h3>Summary</h3>
            <ul>
                <li>Total Rules: {len(self.rules)}</li>
                <li>Passed Rules: {sum(1 for r in self.results if r["success"])}</li>
                <li>Failed Rules: {sum(1 for r in self.results if not r["success"])}</li>
                <li>Critical Failures: {sum(1 for r in self.results if not r["success"] and r["severity"] == "critical")}</li>
                <li>High Failures: {sum(1 for r in self.results if not r["success"] and r["severity"] == "high")}</li>
            </ul>
        """
        
        # Add detailed results if requested
        if include_details:
            body += """
            <h3>Detailed Results</h3>
            <table>
                <tr>
                    <th>Rule ID</th>
                    <th>Status</th>
                    <th>Severity</th>
                    <th>Message</th>
                </tr>
            """
            
            for result in self.results:
                status_class = "pass" if result["success"] else "fail"
                status_text = "PASS" if result["success"] else "FAIL"
                body += f"""
                <tr>
                    <td>{result["rule_id"]}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result["severity"]}</td>
                    <td>{result["message"]}</td>
                </tr>
                """
            
            body += "</table>"
        
        body += """
        </body>
        </html>
        """
        
        msg.set_content("Data Quality Validation Report - Please view in HTML format")
        msg.add_alternative(body, subtype='html')
        
        # Send email
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                if username and password:
                    server.login(username, password)
                server.send_message(msg)
            
            logger.info(f"Sent validation report to {', '.join(email_to)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
            return False


class DataQualityFramework:
    """
    Main framework class to manage data quality rules and validation.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the data quality framework.
        
        Args:
            config_path: Path to configuration file
        """
        self.rules = {}
        self.rule_groups = {}
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Load framework configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {config_path}")
            
            # Process rules from config
            if "rules" in config:
                for rule_config in config["rules"]:
                    # Assuming rule_config is a dictionary with necessary fields
                    # Here you would instantiate your rules based on the configuration
                    # Example:
                    # rule = CompletenessRule(**rule_config)
                    # self.add_rule(rule)
                    pass  # Replace with actual rule processing logic

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the rule to a dictionary representation.
        
        Returns:
            Dictionary representation of the rule
        """
        rule_dict = super().to_dict()
        rule_dict.update({
            "column": self.column,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "include_min": self.include_min,
            "include_max": self.include_max
        })
        return rule_dict

    def validate_data(self, data: pd.DataFrame, group_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Validate the provided data against the defined rules.
        
        Args:
            data: The dataset to validate
            group_name: Optional name of the rule group to validate against
            **kwargs: Additional parameters for validation
        
        Returns:
            A dictionary containing validation results
        """
        # Initialize results
        results = []
        summary = {
            "total_rules": 0,
            "passed_rules": 0,
            "failed_rules": 0,
            "overall_status": "PASS"
        }

        # Determine which rules to apply
        rules_to_apply = self.rules
        if group_name and group_name in self.rule_groups:
            rules_to_apply = [self.rules[rule_id] for rule_id in self.rule_groups[group_name]]

        # Validate each rule
        for rule in rules_to_apply:
            summary["total_rules"] += 1
            result = rule.evaluate(data, **kwargs)
            results.append(result)

            if result["success"]:
                summary["passed_rules"] += 1
            else:
                summary["failed_rules"] += 1

        # Determine overall status
        if summary["failed_rules"] > 0:
            summary["overall_status"] = "FAIL"

        return {
            "summary": summary,
            "results": results
        }


class PatternMatchRule(DataQualityRule):
    """
    Rule to check if values in a column match a specified regex pattern.
    """
    
    def __init__(self, rule_id: str, column: str, pattern: str, 
                match_type: str = "contains", threshold: float = 1.0,
                description: str = None, severity: str = "medium"):
        """
        Initialize a pattern match rule.
        
        Args:
            rule_id: Unique identifier for the rule
            column: Column name to check
            pattern: Regular expression pattern to match
            match_type: Type of match - 'contains', 'fullmatch', or 'exact'
            threshold: Minimum acceptable match rate (0.0 to 1.0)
            description: Human-readable description (auto-generated if None)
            severity: Impact level if rule fails
        """
        if description is None:
            threshold_pct = threshold * 100
            description = (f"Check that {threshold_pct:.1f}% of values in column '{column}' "
                          f"match pattern '{pattern}'")
        
        super().__init__(rule_id, description, severity)
        self.column = column
        self.pattern = pattern
        self.match_type = match_type.lower()
        self.threshold = threshold
        
        # Validate match_type
        valid_match_types = ["contains", "fullmatch", "exact"]
        if self.match_type not in valid_match_types:
            logger.warning(f"Invalid match_type '{match_type}' for rule {rule_id}, "
                         f"using 'contains' instead. Valid options: {valid_match_types}")
            self.match_type = "contains"
        
        # Compile regex pattern
        try:
            self.regex = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {str(e)}")
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Evaluate if values match the specified pattern.
        
        Args:
            data: The dataset to check
            **kwargs: Additional parameters (unused)
            
        Returns:
            Evaluation results
        """
        # Verify column exists
        if self.column not in data.columns:
            return {
                "success": False,
                "message": f"Column '{self.column}' not found in dataframe",
                "metrics": {},
                "details": {"missing_column": self.column}
            }
        
        # Filter out nulls and convert to string
        column_data = data[self.column].dropna().astype(str)
        
        # Handle empty column case
        if len(column_data) == 0:
            return {
                "success": False,
                "message": f"No data to check pattern (column '{self.column}' is empty or all null)",
                "metrics": {
                    "match_rate": 0.0,
                    "non_matching_count": 0
                },
                "details": {
                    "total_values": 0,
                    "null_values": len(data) - len(column_data)
                }
            }
        
        # Check pattern match based on match_type
        if self.match_type == "contains":
            match_mask = column_data.apply(lambda x: bool(self.regex.search(x)))
        elif self.match_type == "fullmatch":
            match_mask = column_data.apply(lambda x: bool(self.regex.fullmatch(x)))
        else:  # exact
            match_mask = column_data.apply(lambda x: x == self.pattern)
        
        # Calculate metrics
        total_values = len(column_data)
        matching_count = match_mask.sum()
        non_matching_count = total_values - matching_count
        match_rate = matching_count / total_values
        
        # Determine success
        success = match_rate >= self.threshold
        
        # Create message
        if success:
            message = (f"Pattern match check passed with rate {match_rate:.2%} "
                      f"meeting threshold {self.threshold:.2%}")
        else:
            message = (f"Pattern match check failed with rate {match_rate:.2%} "
                      f"below threshold {self.threshold:.2%}")
        
        # Initialize non_matching_examples
        non_matching_examples = []

        if non_matching_count > 0:
            non_matching_values = column_data[~match_mask]
            non_matching_examples = non_matching_values.head(5).tolist()
        
        return {
            "success": success,
            "message": message,
            "metrics": {
                "match_rate": match_rate,
                "non_matching_count": int(non_matching_count)
            },
            "details": {
                "total_values": total_values,
                "matching_count": int(matching_count),
                "pattern": self.pattern,
                "match_type": self.match_type,
                "threshold": self.threshold,
                "non_matching_examples": non_matching_examples
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the rule to a dictionary representation.
        
        Returns:
            Dictionary representation of the rule
        """
        rule_dict = super().to_dict()
        rule_dict.update({
            "column": self.column,
            "pattern": self.pattern,
            "match_type": self.match_type,
            "threshold": self.threshold
        })
        return rule_dict


class ReferentialIntegrityRule(DataQualityRule):
    """
    Rule to check for referential integrity between datasets.
    """
    
    def __init__(self, rule_id: str, column: str, reference_data: Union[pd.DataFrame, str],
                reference_column: str = None, ignore_case: bool = False,
                threshold: float = 1.0, description: str = None, 
                severity: str = "medium"):
        """
        Initialize a referential integrity rule.
        
        Args:
            rule_id: Unique identifier for the rule
            column: Column name to check
            reference_data: Reference DataFrame or table name
            reference_column: Column name in reference data (defaults to same as column)
            ignore_case: Whether to ignore case in string comparisons
            threshold: Minimum acceptable match rate (0.0 to 1.0)
            description: Human-readable description (auto-generated if None)
            severity: Impact level if rule fails
        """
        # If reference_column not specified, use the same column name
        if reference_column is None:
            reference_column = column
        
        if description is None:
            # Build description
            ref_name = (reference_data if isinstance(reference_data, str) 
                      else "reference dataset")
            threshold_pct = threshold * 100
            description = (f"Check that {threshold_pct:.1f}% of values in column '{column}' "
                          f"exist in {ref_name}.{reference_column}")
        
        super().__init__(rule_id, description, severity)
        self.column = column
        self.reference_data = reference_data
        self.reference_column = reference_column
        self.ignore_case = ignore_case
        self.threshold = threshold
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Evaluate referential integrity.
        
        Args:
            data: The dataset to check
            **kwargs: Additional parameters, including connection for database references
            
        Returns:
            Evaluation results
        """
        # Verify column exists
        if self.column not in data.columns:
            return {
                "success": False,
                "message": f"Column '{self.column}' not found in dataframe",
                "metrics": {},
                "details": {"missing_column": self.column}
            }
        
        # Get reference data
        if isinstance(self.reference_data, str):
            # Reference is a table name, try to get it from kwargs
            connection = kwargs.get("connection")
            if connection is None:
                return {
                    "success": False,
                    "message": "Database connection required for referential integrity check",
                    "metrics": {},
                    "details": {"reference_table": self.reference_data}
                }
            
            try:
                query = f"SELECT DISTINCT {self.reference_column} FROM {self.reference_data}"
                reference_values = pd.read_sql(query, connection)[self.reference_column]
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error accessing reference table: {str(e)}",
                    "metrics": {},
                    "details": {"reference_table": self.reference_data, "error": str(e)}
                }
        else:
            # Reference is a DataFrame
            if self.reference_column not in self.reference_data.columns:
                return {
                    "success": False,
                    "message": f"Reference column '{self.reference_column}' not found",
                    "metrics": {},
                    "details": {"missing_reference_column": self.reference_column}
                }
            
            reference_values = self.reference_data[self.reference_column].dropna().unique()
        
        # Filter out nulls from data
        column_data = data[self.column].dropna()
        
        # Handle empty column case
        if len(column_data) == 0:
            return {
                "success": False,
                "message": f"No data to check integrity (column '{self.column}' is empty or all null)",
                "metrics": {
                    "match_rate": 0.0,
                    "non_matching_count": 0
                },
                "details": {
                    "total_values": 0,
                    "null_values": len(data) - len(column_data)
                }
            }
        
        # Create sets for efficient lookup
        if self.ignore_case and column_data.dtype == 'object':
            # Convert to lowercase for case-insensitive comparison
            column_set = set(v.lower() if isinstance(v, str) else v for v in column_data)
            reference_set = set(v.lower() if isinstance(v, str) else v for v in reference_values)
        else:
            column_set = set(column_data)
            reference_set = set(reference_values)
        
        # Find non-matching values
        non_matching_values = column_set - reference_set
        
        # Calculate metrics
        total_unique_values = len(column_set)
        matching_unique_count = total_unique_values - len(non_matching_values)
        
        # Count occurrences of non-matching values in original data
        if non_matching_values:
            if self.ignore_case and column_data.dtype == 'object':
                # Case-insensitive check
                non_matching_mask = column_data.apply(
                    lambda x: x.lower() if isinstance(x, str) else x
                ).isin(non_matching_values)
            else:
                non_matching_mask = column_data.isin(non_matching_values)
            
            non_matching_count = non_matching_mask.sum()
        else:
            non_matching_count = 0
        
        total_values = len(column_data)
        match_rate = (total_values - non_matching_count) / total_values
        
        # Determine success
        success = match_rate >= self.threshold
        
        # Create message
        if success:
            message = (f"Referential integrity check passed with rate {match_rate:.2%} "
                      f"meeting threshold {self.threshold:.2%}")
        else:
            message = (f"Referential integrity check failed with rate {match_rate:.2%} "
                      f"below threshold {self.threshold:.2%}")
        
        # Initialize non_matching_examples
        non_matching_examples = []

        if non_matching_values:
            # Convert set to list for examples
            non_matching_list = list(non_matching_values)
            non_matching_examples = non_matching_list[:5]  # First 5 examples
        
        return {
            "success": success,
            "message": message,
            "metrics": {
                "match_rate": match_rate,
                "non_matching_count": int(non_matching_count),
                "unique_values_count": total_unique_values,
                "matching_unique_count": matching_unique_count
            },
            "details": {
                "total_values": total_values,
                "non_matching_examples": non_matching_examples
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the rule to a dictionary representation.
        
        Returns:
            Dictionary representation of the rule
        """
        rule_dict = super().to_dict()
        rule_dict.update({
            "column": self.column,
            "reference_data": str(self.reference_data),
            "reference_column": self.reference_column,
            "ignore_case": self.ignore_case,
            "threshold": self.threshold
        })
        return rule_dict