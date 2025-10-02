"""
Data Profiling Tool for Univariate Analysis

This tool provides comprehensive data profiling capabilities for target variables
using statistical analysis and data quality assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
from dataclasses import dataclass
from enum import Enum

class DataType(Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"
    TEMPORAL = "temporal"

@dataclass
class StatisticalSummary:
    """Statistical summary for a variable"""
    count: int
    missing: int
    missing_percentage: float
    unique_values: int
    data_type: DataType
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    mode: Optional[Any] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

class DataProfilingTool:
    """Tool for comprehensive data profiling of variables"""
    
    def __init__(self):
        """Initialize the data profiling tool"""
        self.logger = logging.getLogger(__name__)
    
    def is_ready(self) -> bool:
        """Check if the data profiling tool is ready to use"""
        return True
    
    def profile_variable(self, variable_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive profile for a variable
        
        Args:
            variable_id: ID of the variable to profile (e.g., "table.column")
            data: DataFrame containing the data
            
        Returns:
            Dictionary containing comprehensive variable profile
        """
        try:
            if data is None or data.empty:
                return self._create_empty_profile(variable_id)
            
            # Extract column name from variable_id
            column_name = variable_id.split('.')[-1] if '.' in variable_id else variable_id
            
            # Get the actual column data
            if column_name not in data.columns:
                self.logger.error(f"Column {column_name} not found in data")
                return self._create_empty_profile(variable_id)
            
            column_data = data[column_name]
            
            # Generate statistical summary
            statistical_summary = self._generate_statistical_summary(column_data, variable_id)
            
            # Determine data type
            data_type = self._determine_data_type(column_data)
            
            # Generate distribution analysis
            distribution = self._analyze_distribution(column_data, data_type)
            
            # Generate completeness analysis
            completeness = self._analyze_completeness(column_data)
            
            # Generate uniqueness analysis
            uniqueness = self._analyze_uniqueness(column_data)
            
            return {
                "variable_id": variable_id,
                "column_name": column_name,
                "data_type": data_type.value,
                "statistical_summary": statistical_summary.__dict__,
                "distribution": distribution,
                "completeness": completeness,
                "uniqueness": uniqueness,
                "profile_timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Data profiling failed for {variable_id}: {str(e)}")
            return self._create_error_profile(variable_id, str(e))
    
    
    def _generate_statistical_summary(self, column_data: pd.Series, variable_id: str) -> StatisticalSummary:
        """Generate comprehensive statistical summary"""
        try:
            # Basic counts
            count = len(column_data)
            missing = column_data.isnull().sum()
            missing_percentage = (missing / count * 100) if count > 0 else 0
            unique_values = column_data.nunique()
            
            # Determine data type
            data_type = self._determine_data_type(column_data)
            
            # Initialize summary
            summary = StatisticalSummary(
                count=count,
                missing=missing,
                missing_percentage=missing_percentage,
                unique_values=unique_values,
                data_type=data_type
            )
            
            # Calculate statistics based on data type
            if data_type in [DataType.CONTINUOUS, DataType.ORDINAL]:
                # Numerical statistics
                summary.mean = column_data.mean()
                summary.median = column_data.median()
                summary.std = column_data.std()
                summary.min = column_data.min()
                summary.max = column_data.max()
                summary.q25 = column_data.quantile(0.25)
                summary.q75 = column_data.quantile(0.75)
                
                # Advanced statistics
                try:
                    summary.skewness = column_data.skew()
                    summary.kurtosis = column_data.kurtosis()
                except:
                    summary.skewness = None
                    summary.kurtosis = None
            
            elif data_type in [DataType.CATEGORICAL, DataType.BINARY]:
                # Categorical statistics
                summary.mode = column_data.mode().iloc[0] if not column_data.mode().empty else None
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Statistical summary generation failed: {str(e)}")
            return StatisticalSummary(
                count=0,
                missing=0,
                missing_percentage=0,
                unique_values=0,
                data_type=DataType.CATEGORICAL
            )
    
    def _determine_data_type(self, column_data: pd.Series) -> DataType:
        """Determine the data type of a column"""
        try:
            # Remove null values for type detection
            clean_data = column_data.dropna()
            
            if len(clean_data) == 0:
                return DataType.CATEGORICAL
            
            # Check if it's numeric
            if pd.api.types.is_numeric_dtype(clean_data):
                # Check if it's binary
                unique_values = clean_data.nunique()
                if unique_values == 2:
                    return DataType.BINARY
                # Check if it's ordinal (small number of unique values)
                elif unique_values <= 10:
                    return DataType.ORDINAL
                else:
                    return DataType.CONTINUOUS
            
            # Check if it's datetime
            elif pd.api.types.is_datetime64_any_dtype(clean_data):
                return DataType.TEMPORAL
            
            # Check if it's categorical
            else:
                return DataType.CATEGORICAL
                
        except Exception as e:
            self.logger.error(f"Data type determination failed: {str(e)}")
            return DataType.CATEGORICAL
    
    def _analyze_distribution(self, column_data: pd.Series, data_type: DataType) -> Dict[str, Any]:
        """Analyze the distribution of the data"""
        try:
            distribution = {
                "type": data_type.value,
                "shape": "unknown",
                "outliers_present": False,
                "distribution_characteristics": []
            }
            
            if data_type in [DataType.CONTINUOUS, DataType.ORDINAL]:
                # Analyze numerical distribution
                clean_data = column_data.dropna()
                
                if len(clean_data) > 0:
                    # Check for skewness
                    skewness = clean_data.skew()
                    if abs(skewness) > 1:
                        distribution["shape"] = "highly_skewed"
                        distribution["distribution_characteristics"].append(f"Skewness: {skewness:.2f}")
                    elif abs(skewness) > 0.5:
                        distribution["shape"] = "moderately_skewed"
                        distribution["distribution_characteristics"].append(f"Skewness: {skewness:.2f}")
                    else:
                        distribution["shape"] = "approximately_normal"
                    
                    # Check for outliers using IQR method
                    q25 = clean_data.quantile(0.25)
                    q75 = clean_data.quantile(0.75)
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    
                    outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                    if len(outliers) > 0:
                        distribution["outliers_present"] = True
                        distribution["distribution_characteristics"].append(f"Outliers: {len(outliers)}")
            
            elif data_type == DataType.CATEGORICAL:
                # Analyze categorical distribution
                value_counts = column_data.value_counts()
                total_values = len(column_data.dropna())
                
                if total_values > 0:
                    # Check for high cardinality
                    if len(value_counts) > total_values * 0.5:
                        distribution["shape"] = "high_cardinality"
                        distribution["distribution_characteristics"].append("High cardinality")
                    else:
                        distribution["shape"] = "low_cardinality"
                        distribution["distribution_characteristics"].append("Low cardinality")
                    
                    # Check for imbalanced classes
                    max_frequency = value_counts.max()
                    if max_frequency > total_values * 0.8:
                        distribution["distribution_characteristics"].append("Highly imbalanced")
                    elif max_frequency > total_values * 0.6:
                        distribution["distribution_characteristics"].append("Moderately imbalanced")
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Distribution analysis failed: {str(e)}")
            return {"type": data_type.value, "shape": "unknown", "error": str(e)}
    
    def _analyze_completeness(self, column_data: pd.Series) -> Dict[str, Any]:
        """Analyze data completeness"""
        try:
            total_count = len(column_data)
            missing_count = column_data.isnull().sum()
            complete_count = total_count - missing_count
            
            completeness = {
                "total_count": total_count,
                "complete_count": complete_count,
                "missing_count": missing_count,
                "completeness_percentage": (complete_count / total_count * 100) if total_count > 0 else 0,
                "missing_percentage": (missing_count / total_count * 100) if total_count > 0 else 0,
                "completeness_quality": "unknown"
            }
            
            # Determine completeness quality
            if completeness["completeness_percentage"] >= 95:
                completeness["completeness_quality"] = "excellent"
            elif completeness["completeness_percentage"] >= 90:
                completeness["completeness_quality"] = "good"
            elif completeness["completeness_percentage"] >= 80:
                completeness["completeness_quality"] = "fair"
            else:
                completeness["completeness_quality"] = "poor"
            
            return completeness
            
        except Exception as e:
            self.logger.error(f"Completeness analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_uniqueness(self, column_data: pd.Series) -> Dict[str, Any]:
        """Analyze data uniqueness"""
        try:
            total_count = len(column_data)
            unique_count = column_data.nunique()
            duplicate_count = total_count - unique_count
            
            uniqueness = {
                "total_count": total_count,
                "unique_count": unique_count,
                "duplicate_count": duplicate_count,
                "uniqueness_percentage": (unique_count / total_count * 100) if total_count > 0 else 0,
                "duplicate_percentage": (duplicate_count / total_count * 100) if total_count > 0 else 0,
                "uniqueness_quality": "unknown"
            }
            
            # Determine uniqueness quality
            if uniqueness["uniqueness_percentage"] >= 95:
                uniqueness["uniqueness_quality"] = "highly_unique"
            elif uniqueness["uniqueness_percentage"] >= 80:
                uniqueness["uniqueness_quality"] = "moderately_unique"
            elif uniqueness["uniqueness_percentage"] >= 50:
                uniqueness["uniqueness_quality"] = "low_uniqueness"
            else:
                uniqueness["uniqueness_quality"] = "very_low_uniqueness"
            
            return uniqueness
            
        except Exception as e:
            self.logger.error(f"Uniqueness analysis failed: {str(e)}")
            return {"error": str(e)}
    
    
    def _create_empty_profile(self, variable_id: str) -> Dict[str, Any]:
        """Create empty profile for missing data"""
        return {
            "variable_id": variable_id,
            "column_name": variable_id.split('.')[-1] if '.' in variable_id else variable_id,
            "data_type": "unknown",
            "statistical_summary": {
                "count": 0,
                "missing": 0,
                "missing_percentage": 0,
                "unique_values": 0,
                "data_type": "unknown"
            },
            "distribution": {"type": "unknown", "shape": "unknown"},
            "completeness": {"completeness_quality": "unknown"},
            "uniqueness": {"uniqueness_quality": "unknown"},
            "metadata": {},
            "error": "No data available"
        }
    
    def _create_error_profile(self, variable_id: str, error_message: str) -> Dict[str, Any]:
        """Create error profile"""
        return {
            "variable_id": variable_id,
            "column_name": variable_id.split('.')[-1] if '.' in variable_id else variable_id,
            "data_type": "unknown",
            "error": error_message,
            "profile_timestamp": pd.Timestamp.now().isoformat()
        }
    
    def profile_multiple_variables(self, variable_ids: List[str], data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Profile multiple variables in batch
        
        Args:
            variable_ids: List of variable IDs to profile
            data: DataFrame containing the data
            
        Returns:
            Dictionary mapping variable IDs to their profiles
        """
        profiles = {}
        
        for variable_id in variable_ids:
            try:
                profiles[variable_id] = self.profile_variable(variable_id, data)
            except Exception as e:
                self.logger.error(f"Failed to profile {variable_id}: {str(e)}")
                profiles[variable_id] = self._create_error_profile(variable_id, str(e))
        
        return profiles


# Global instance for easy access
data_profiling_tool = DataProfilingTool()

# Convenience functions for direct use
def profile_variable(variable_id: str, data: pd.DataFrame) -> Dict[str, Any]:
    """Profile a single variable"""
    return data_profiling_tool.profile_variable(variable_id, data)

def profile_multiple_variables(variable_ids: List[str], data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Profile multiple variables in batch"""
    return data_profiling_tool.profile_multiple_variables(variable_ids, data)

def is_data_profiling_ready() -> bool:
    """Check if data profiling tool is ready to use"""
    return data_profiling_tool.is_ready()
