"""
Anomaly Detection Tool for Univariate Analysis

This tool provides comprehensive anomaly detection capabilities for target variables
using various statistical and machine learning methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyMethod(Enum):
    IQR = "iqr"
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"

@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    method: AnomalyMethod
    outliers_count: int
    outliers_percentage: float
    outlier_indices: List[int]
    outlier_values: List[Any]
    anomaly_scores: Optional[List[float]] = None
    thresholds: Optional[Dict[str, float]] = None
    confidence_level: float = 0.95

class AnomalyDetectionTool:
    """Tool for detecting anomalies and outliers in variables"""
    
    def __init__(self, kb=None, data_product_builder=None):
        """
        Initialize the anomaly detection tool
        
        Args:
            kb: Intugle KnowledgeBuilder instance
            data_product_builder: Intugle DataProductBuilder instance
        """
        self.kb = kb
        self.data_product_builder = data_product_builder
        self.logger = logging.getLogger(__name__)
    
    def detect_anomalies(self, 
                        variable_id: str, 
                        data: Optional[pd.DataFrame] = None,
                        methods: List[AnomalyMethod] = None,
                        confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Detect anomalies in a variable using multiple methods
        
        Args:
            variable_id: ID of the variable to analyze
            data: Optional DataFrame containing the data
            methods: List of anomaly detection methods to use
            confidence_level: Confidence level for statistical methods
            
        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            # Default methods if not specified
            if methods is None:
                methods = [AnomalyMethod.IQR, AnomalyMethod.Z_SCORE, AnomalyMethod.ISOLATION_FOREST]
            
            # Get data if not provided
            if data is None:
                data = self._fetch_variable_data(variable_id)
            
            if data is None or data.empty:
                return self._create_empty_anomaly_result(variable_id)
            
            # Extract column name from variable_id
            column_name = variable_id.split('.')[-1] if '.' in variable_id else variable_id
            
            if column_name not in data.columns:
                self.logger.error(f"Column {column_name} not found in data")
                return self._create_empty_anomaly_result(variable_id)
            
            column_data = data[column_name]
            
            # Detect anomalies using each method
            anomaly_results = {}
            for method in methods:
                try:
                    result = self._detect_anomalies_method(column_data, method, confidence_level)
                    anomaly_results[method.value] = result
                except Exception as e:
                    self.logger.error(f"Anomaly detection failed for method {method.value}: {str(e)}")
                    anomaly_results[method.value] = self._create_error_result(method, str(e))
            
            # Combine results from all methods
            combined_result = self._combine_anomaly_results(anomaly_results)
            
            # Analyze missing data patterns
            missing_patterns = self._analyze_missing_patterns(column_data)
            
            # Detect data quality issues
            quality_issues = self._detect_data_quality_issues(column_data)
            
            return {
                "variable_id": variable_id,
                "column_name": column_name,
                "anomaly_results": anomaly_results,
                "combined_result": combined_result,
                "missing_patterns": missing_patterns,
                "data_quality_issues": quality_issues,
                "detection_timestamp": pd.Timestamp.now().isoformat(),
                "confidence_level": confidence_level
            }
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed for {variable_id}: {str(e)}")
            return self._create_error_anomaly_result(variable_id, str(e))
    
    def _detect_anomalies_method(self, 
                                column_data: pd.Series, 
                                method: AnomalyMethod, 
                                confidence_level: float) -> AnomalyResult:
        """Detect anomalies using a specific method"""
        
        if method == AnomalyMethod.IQR:
            return self._detect_iqr_anomalies(column_data)
        elif method == AnomalyMethod.Z_SCORE:
            return self._detect_z_score_anomalies(column_data, confidence_level)
        elif method == AnomalyMethod.MODIFIED_Z_SCORE:
            return self._detect_modified_z_score_anomalies(column_data)
        elif method == AnomalyMethod.ISOLATION_FOREST:
            return self._detect_isolation_forest_anomalies(column_data)
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")
    
    def _detect_iqr_anomalies(self, column_data: pd.Series) -> AnomalyResult:
        """Detect anomalies using Interquartile Range (IQR) method"""
        try:
            # Remove null values
            clean_data = column_data.dropna()
            
            if len(clean_data) == 0:
                return AnomalyResult(
                    method=AnomalyMethod.IQR,
                    outliers_count=0,
                    outliers_percentage=0.0,
                    outlier_indices=[],
                    outlier_values=[],
                    thresholds={"q25": None, "q75": None, "iqr": None}
                )
            
            # Calculate quartiles
            q25 = clean_data.quantile(0.25)
            q75 = clean_data.quantile(0.75)
            iqr = q75 - q25
            
            # Define outlier bounds
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Find outliers
            outlier_mask = (clean_data < lower_bound) | (clean_data > upper_bound)
            outlier_indices = clean_data[outlier_mask].index.tolist()
            outlier_values = clean_data[outlier_mask].tolist()
            
            return AnomalyResult(
                method=AnomalyMethod.IQR,
                outliers_count=len(outlier_indices),
                outliers_percentage=(len(outlier_indices) / len(clean_data) * 100),
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                thresholds={
                    "q25": q25,
                    "q75": q75,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound
                }
            )
            
        except Exception as e:
            self.logger.error(f"IQR anomaly detection failed: {str(e)}")
            return AnomalyResult(
                method=AnomalyMethod.IQR,
                outliers_count=0,
                outliers_percentage=0.0,
                outlier_indices=[],
                outlier_values=[],
                error=str(e)
            )
    
    def _detect_z_score_anomalies(self, column_data: pd.Series, confidence_level: float) -> AnomalyResult:
        """Detect anomalies using Z-score method"""
        try:
            # Remove null values
            clean_data = column_data.dropna()
            
            if len(clean_data) == 0:
                return AnomalyResult(
                    method=AnomalyMethod.Z_SCORE,
                    outliers_count=0,
                    outliers_percentage=0.0,
                    outlier_indices=[],
                    outlier_values=[],
                    thresholds={"z_threshold": None}
                )
            
            # Calculate Z-scores
            mean = clean_data.mean()
            std = clean_data.std()
            
            if std == 0:
                # No variation in data
                return AnomalyResult(
                    method=AnomalyMethod.Z_SCORE,
                    outliers_count=0,
                    outliers_percentage=0.0,
                    outlier_indices=[],
                    outlier_values=[],
                    thresholds={"z_threshold": None, "mean": mean, "std": std}
                )
            
            z_scores = np.abs((clean_data - mean) / std)
            
            # Determine threshold based on confidence level
            if confidence_level == 0.95:
                z_threshold = 1.96
            elif confidence_level == 0.99:
                z_threshold = 2.58
            elif confidence_level == 0.999:
                z_threshold = 3.29
            else:
                # Default to 3 standard deviations
                z_threshold = 3.0
            
            # Find outliers
            outlier_mask = z_scores > z_threshold
            outlier_indices = clean_data[outlier_mask].index.tolist()
            outlier_values = clean_data[outlier_mask].tolist()
            anomaly_scores = z_scores[outlier_mask].tolist()
            
            return AnomalyResult(
                method=AnomalyMethod.Z_SCORE,
                outliers_count=len(outlier_indices),
                outliers_percentage=(len(outlier_indices) / len(clean_data) * 100),
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                anomaly_scores=anomaly_scores,
                thresholds={
                    "z_threshold": z_threshold,
                    "mean": mean,
                    "std": std
                },
                confidence_level=confidence_level
            )
            
        except Exception as e:
            self.logger.error(f"Z-score anomaly detection failed: {str(e)}")
            return AnomalyResult(
                method=AnomalyMethod.Z_SCORE,
                outliers_count=0,
                outliers_percentage=0.0,
                outlier_indices=[],
                outlier_values=[],
                error=str(e)
            )
    
    def _detect_modified_z_score_anomalies(self, column_data: pd.Series) -> AnomalyResult:
        """Detect anomalies using Modified Z-score method (using median)"""
        try:
            # Remove null values
            clean_data = column_data.dropna()
            
            if len(clean_data) == 0:
                return AnomalyResult(
                    method=AnomalyMethod.MODIFIED_Z_SCORE,
                    outliers_count=0,
                    outliers_percentage=0.0,
                    outlier_indices=[],
                    outlier_values=[],
                    thresholds={"modified_z_threshold": None}
                )
            
            # Calculate median and median absolute deviation
            median = clean_data.median()
            mad = np.median(np.abs(clean_data - median))
            
            if mad == 0:
                # No variation in data
                return AnomalyResult(
                    method=AnomalyMethod.MODIFIED_Z_SCORE,
                    outliers_count=0,
                    outliers_percentage=0.0,
                    outlier_indices=[],
                    outlier_values=[],
                    thresholds={"modified_z_threshold": None, "median": median, "mad": mad}
                )
            
            # Calculate modified Z-scores
            modified_z_scores = 0.6745 * (clean_data - median) / mad
            modified_z_scores = np.abs(modified_z_scores)
            
            # Use threshold of 3.5 for modified Z-score
            modified_z_threshold = 3.5
            
            # Find outliers
            outlier_mask = modified_z_scores > modified_z_threshold
            outlier_indices = clean_data[outlier_mask].index.tolist()
            outlier_values = clean_data[outlier_mask].tolist()
            anomaly_scores = modified_z_scores[outlier_mask].tolist()
            
            return AnomalyResult(
                method=AnomalyMethod.MODIFIED_Z_SCORE,
                outliers_count=len(outlier_indices),
                outliers_percentage=(len(outlier_indices) / len(clean_data) * 100),
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                anomaly_scores=anomaly_scores,
                thresholds={
                    "modified_z_threshold": modified_z_threshold,
                    "median": median,
                    "mad": mad
                }
            )
            
        except Exception as e:
            self.logger.error(f"Modified Z-score anomaly detection failed: {str(e)}")
            return AnomalyResult(
                method=AnomalyMethod.MODIFIED_Z_SCORE,
                outliers_count=0,
                outliers_percentage=0.0,
                outlier_indices=[],
                outlier_values=[],
                error=str(e)
            )
    
    def _detect_isolation_forest_anomalies(self, column_data: pd.Series) -> AnomalyResult:
        """Detect anomalies using Isolation Forest method"""
        try:
            # Remove null values
            clean_data = column_data.dropna()
            
            if len(clean_data) == 0:
                return AnomalyResult(
                    method=AnomalyMethod.ISOLATION_FOREST,
                    outliers_count=0,
                    outliers_percentage=0.0,
                    outlier_indices=[],
                    outlier_values=[],
                    thresholds={"contamination": None}
                )
            
            # Convert to numeric if possible
            if not pd.api.types.is_numeric_dtype(clean_data):
                # For non-numeric data, use label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                numeric_data = le.fit_transform(clean_data.astype(str))
            else:
                numeric_data = clean_data.values
            
            # Reshape for sklearn
            X = numeric_data.reshape(-1, 1)
            
            # Apply Isolation Forest
            contamination = 0.1  # Assume 10% contamination
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            anomaly_scores = iso_forest.score_samples(X)
            
            # Find outliers (label -1)
            outlier_mask = outlier_labels == -1
            outlier_indices = clean_data[outlier_mask].index.tolist()
            outlier_values = clean_data[outlier_mask].tolist()
            outlier_scores = anomaly_scores[outlier_mask].tolist()
            
            return AnomalyResult(
                method=AnomalyMethod.ISOLATION_FOREST,
                outliers_count=len(outlier_indices),
                outliers_percentage=(len(outlier_indices) / len(clean_data) * 100),
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                anomaly_scores=outlier_scores,
                thresholds={"contamination": contamination}
            )
            
        except Exception as e:
            self.logger.error(f"Isolation Forest anomaly detection failed: {str(e)}")
            return AnomalyResult(
                method=AnomalyMethod.ISOLATION_FOREST,
                outliers_count=0,
                outliers_percentage=0.0,
                outlier_indices=[],
                outlier_values=[],
                error=str(e)
            )
    
    def _combine_anomaly_results(self, anomaly_results: Dict[str, AnomalyResult]) -> Dict[str, Any]:
        """Combine results from multiple anomaly detection methods"""
        try:
            # Get all outlier indices
            all_outlier_indices = set()
            method_agreement = {}
            
            for method_name, result in anomaly_results.items():
                if hasattr(result, 'outlier_indices'):
                    all_outlier_indices.update(result.outlier_indices)
            
            # Calculate agreement between methods
            for method_name, result in anomaly_results.items():
                if hasattr(result, 'outlier_indices'):
                    method_outliers = set(result.outlier_indices)
                    agreement_count = len(method_outliers.intersection(all_outlier_indices))
                    method_agreement[method_name] = {
                        "outliers_found": len(method_outliers),
                        "agreement_count": agreement_count,
                        "agreement_percentage": (agreement_count / len(all_outlier_indices) * 100) if all_outlier_indices else 0
                    }
            
            # Find consensus outliers (detected by multiple methods)
            consensus_outliers = set()
            for method_name, result in anomaly_results.items():
                if hasattr(result, 'outlier_indices'):
                    if len(consensus_outliers) == 0:
                        consensus_outliers = set(result.outlier_indices)
                    else:
                        consensus_outliers = consensus_outliers.intersection(set(result.outlier_indices))
            
            return {
                "total_unique_outliers": len(all_outlier_indices),
                "consensus_outliers": len(consensus_outliers),
                "consensus_outlier_indices": list(consensus_outliers),
                "method_agreement": method_agreement,
                "recommended_action": self._get_anomaly_recommendation(anomaly_results)
            }
            
        except Exception as e:
            self.logger.error(f"Anomaly result combination failed: {str(e)}")
            return {"error": str(e)}
    
    def _get_anomaly_recommendation(self, anomaly_results: Dict[str, AnomalyResult]) -> str:
        """Get recommendation based on anomaly detection results"""
        try:
            total_outliers = 0
            high_confidence_outliers = 0
            
            for method_name, result in anomaly_results.items():
                if hasattr(result, 'outliers_count'):
                    total_outliers += result.outliers_count
                    if result.outliers_count > 0 and result.outliers_percentage < 5:
                        high_confidence_outliers += 1
            
            if total_outliers == 0:
                return "No anomalies detected. Data appears clean."
            elif total_outliers < 10:
                return "Few anomalies detected. Consider manual review of outlier values."
            elif total_outliers < 50:
                return "Moderate number of anomalies detected. Review and consider data cleaning."
            else:
                return "High number of anomalies detected. Comprehensive data cleaning recommended."
                
        except Exception as e:
            return f"Unable to generate recommendation: {str(e)}"
    
    def _analyze_missing_patterns(self, column_data: pd.Series) -> Dict[str, Any]:
        """Analyze patterns in missing data"""
        try:
            missing_count = column_data.isnull().sum()
            total_count = len(column_data)
            
            if missing_count == 0:
                return {
                    "pattern": "no_missing",
                    "missing_count": 0,
                    "missing_percentage": 0.0,
                    "description": "No missing values found"
                }
            
            # Check for patterns
            missing_indices = column_data.isnull()
            
            # Check for consecutive missing values
            consecutive_missing = 0
            max_consecutive = 0
            current_consecutive = 0
            
            for is_missing in missing_indices:
                if is_missing:
                    current_consecutive += 1
                    consecutive_missing = max(consecutive_missing, current_consecutive)
                else:
                    max_consecutive = max(max_consecutive, current_consecutive)
                    current_consecutive = 0
            
            max_consecutive = max(max_consecutive, current_consecutive)
            
            # Determine pattern
            if max_consecutive > total_count * 0.1:
                pattern = "block_missing"
                description = f"Large blocks of missing data (max {max_consecutive} consecutive)"
            elif missing_count / total_count > 0.5:
                pattern = "high_missing"
                description = f"High percentage of missing data ({missing_count/total_count*100:.1f}%)"
            else:
                pattern = "random_missing"
                description = f"Random missing data pattern ({missing_count} missing values)"
            
            return {
                "pattern": pattern,
                "missing_count": missing_count,
                "missing_percentage": (missing_count / total_count * 100),
                "max_consecutive_missing": max_consecutive,
                "description": description
            }
            
        except Exception as e:
            self.logger.error(f"Missing pattern analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _detect_data_quality_issues(self, column_data: pd.Series) -> List[Dict[str, Any]]:
        """Detect various data quality issues"""
        issues = []
        
        try:
            # Check for inconsistent data types
            if not pd.api.types.is_numeric_dtype(column_data):
                # Check for mixed types in string columns
                unique_types = set(type(val).__name__ for val in column_data.dropna())
                if len(unique_types) > 1:
                    issues.append({
                        "type": "mixed_data_types",
                        "severity": "medium",
                        "description": f"Mixed data types found: {list(unique_types)}"
                    })
            
            # Check for duplicate values
            duplicate_count = column_data.duplicated().sum()
            if duplicate_count > 0:
                issues.append({
                    "type": "duplicates",
                    "severity": "low",
                    "description": f"{duplicate_count} duplicate values found"
                })
            
            # Check for extreme values (for numeric data)
            if pd.api.types.is_numeric_dtype(column_data):
                clean_data = column_data.dropna()
                if len(clean_data) > 0:
                    # Check for values that are orders of magnitude different
                    q99 = clean_data.quantile(0.99)
                    q01 = clean_data.quantile(0.01)
                    if q99 / q01 > 1000:  # More than 3 orders of magnitude
                        issues.append({
                            "type": "extreme_value_range",
                            "severity": "medium",
                            "description": f"Extreme value range: {q01:.2f} to {q99:.2f}"
                        })
            
            # Check for suspicious patterns in string data
            if not pd.api.types.is_numeric_dtype(column_data):
                clean_data = column_data.dropna().astype(str)
                
                # Check for suspicious patterns
                if len(clean_data) > 0:
                    # Check for values that are mostly numbers but stored as strings
                    numeric_like = clean_data.str.match(r'^\d+\.?\d*$').sum()
                    if numeric_like / len(clean_data) > 0.8:
                        issues.append({
                            "type": "numeric_as_string",
                            "severity": "medium",
                            "description": f"{numeric_like} values appear to be numeric but stored as strings"
                        })
                    
                    # Check for empty strings
                    empty_strings = (clean_data == '').sum()
                    if empty_strings > 0:
                        issues.append({
                            "type": "empty_strings",
                            "severity": "low",
                            "description": f"{empty_strings} empty string values found"
                        })
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Data quality issue detection failed: {str(e)}")
            return [{"type": "detection_error", "severity": "high", "description": str(e)}]
    
    def _fetch_variable_data(self, variable_id: str) -> Optional[pd.DataFrame]:
        """Fetch data for a variable using Intugle DataProductBuilder"""
        try:
            if not self.data_product_builder:
                return None
            
            # Parse table and column from variable_id
            if '.' not in variable_id:
                return None
            
            table_name, column_name = variable_id.split('.', 1)
            
            # Create a simple ETL model to fetch the data
            from intugle.libs.smart_query_generator.models.models import ETLModel, FieldsModel
            
            field = FieldsModel(
                id=column_name,
                name=column_name,
                category="dimension"
            )
            
            etl_model = ETLModel(
                name=f"anomaly_{variable_id.replace('.', '_')}",
                fields=[field]
            )
            
            # Generate and execute query
            dataset = self.data_product_builder.build(etl_model)
            return dataset.to_df()
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {variable_id}: {str(e)}")
            return None
    
    def _create_empty_anomaly_result(self, variable_id: str) -> Dict[str, Any]:
        """Create empty anomaly result for missing data"""
        return {
            "variable_id": variable_id,
            "column_name": variable_id.split('.')[-1] if '.' in variable_id else variable_id,
            "anomaly_results": {},
            "combined_result": {"total_unique_outliers": 0},
            "missing_patterns": {"pattern": "no_data"},
            "data_quality_issues": [],
            "error": "No data available"
        }
    
    def _create_error_anomaly_result(self, variable_id: str, error_message: str) -> Dict[str, Any]:
        """Create error anomaly result"""
        return {
            "variable_id": variable_id,
            "column_name": variable_id.split('.')[-1] if '.' in variable_id else variable_id,
            "error": error_message,
            "detection_timestamp": pd.Timestamp.now().isoformat()
        }
    
    def _create_error_result(self, method: AnomalyMethod, error_message: str) -> AnomalyResult:
        """Create error result for a specific method"""
        return AnomalyResult(
            method=method,
            outliers_count=0,
            outliers_percentage=0.0,
            outlier_indices=[],
            outlier_values=[],
            error=error_message
        )
